/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_X64_IP_CONVOLUTION_HPP
#define CPU_X64_IP_CONVOLUTION_HPP

#include <string>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_inner_product_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {

status_t reshape_dst(memory_desc_t *o_md, const memory_desc_t *i_md) {
    dims_t reduce {};
    const dim_t ndims = 2; // dst is always nc for inner product
    // conv to ip: remove spatial
    for (int d = 0; d < ndims; ++d)
        reduce[d] = i_md->dims[d];

    return dnnl_memory_desc_reshape(o_md, i_md, ndims, reduce);
}

status_t maybe_reshape_weights(memory_desc_t *o_md, const memory_desc_t *i_md,
        bool with_groups, bool to_ip = false) {
    dims_t reduce {};
    const dim_t ndims = i_md->ndims + (to_ip ? -1 : +1) * with_groups;
    if (to_ip) {
        // conv to ip: maybe remove groups
        for (int d = 0; d < ndims; ++d)
            reduce[d] = i_md->dims[d + with_groups];
    } else {
        // ip to conv: maybe restore groups
        if (with_groups) reduce[0] = 1;
        for (int d = 0; d < ndims; ++d)
            reduce[d + with_groups] = i_md->dims[d];
    }

    return dnnl_memory_desc_reshape(o_md, i_md, ndims, reduce);
}

} // namespace

struct ip_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : cpu_convolution_fwd_pd_t(other), ip_pd_(other.ip_pd_->clone()) {}

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(name_.c_str(), ip_convolution_fwd_t);

        status_t init_ip(engine_t *engine) {
            inner_product_desc_t ipd;
            CHECK(ip_desc_create(&ipd));
            dnnl_primitive_desc_iterator it(
                    engine, (op_desc_t *)&ipd, attr(), nullptr);
            if (!it.is_initialized()) return status::out_of_memory;

            while (++it != it.end()) {
                ip_pd_.reset(it.fetch_once());
                const bool ok = ip_pd_->weights_md()->extra.flags == 0;
                if (ok) return status::success;
            }
            return status::unimplemented;
        }

        status_t init(engine_t *engine) {
            using namespace format_tag;
            using smask_t = primitive_attr_t::skip_mask_t;

            // Check if convolution is equivalent to inner product.
            const bool is_ip_applicable = true
                    // no dilations
                    && utils::everyone_is(0, KDD(), KDH(), KDW())
                    // no "left" padding
                    && utils::everyone_is(0, padFront(), padT(), padL())
                    // no "right" padding
                    && utils::everyone_is(0, padBack(), padB(), padR())
                    // no non-trivial groups or output spatial
                    && utils::everyone_is(1, G(), OD(), OH(), OW())
                    // only unit strides
                    && utils::everyone_is(1, KSD(), KSH(), KSW());
            if (!is_ip_applicable) return status::unimplemented;

            // Simple heuristic to only target arches and shapes that benefit.
            // TODO: Extend to other arches and shapes as performance allows.
            const dim_t ks = KD() * KH() * KW();
            const dim_t ks_threshold = 27; // empirical
            const bool is_performant
                    = 1 < MB() && ks > ks_threshold && mayiuse(avx512_core);
            if (!is_performant) return status::unimplemented;

            const bool ok = is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && attr()->has_default_values(
                            smask_t::oscale | smask_t::post_ops);
            if (!ok) return status::unimplemented;

            // Check that nspc is the default layout for convolutions.
            // Otherwise, do not set formats in case of `format_kind::any`.
            // Currently this means:
            //  - int8 with any forward prop_kind on any isa
            //  - fp32 with `forward_inference` on avx512_core and higher
            // TODO: Add support for bf16 inference on avx512_core and higher.
            const bool set_any_to_nspc = false
                    || (weights_md_.data_type == data_type::f32
                            && desc()->prop_kind == prop_kind::forward_inference
                            && mayiuse(avx512_core))
                    || weights_md_.data_type == data_type::s8;
            CHECK(set_and_or_check_formats(set_any_to_nspc));

            CHECK(init_ip(engine));

            if (weights_md_.format_kind == format_kind::any)
                CHECK(maybe_reshape_weights(
                        &weights_md_, ip_pd_->weights_md(), with_groups()));

            init_name();
            init_scratchpad();
            return status::success;
        }

        std::unique_ptr<primitive_desc_t> ip_pd_;

    private:
        std::string name_ = "ip:";

        void init_name() { name_.append(ip_pd_->name()); }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_nested, ip_pd_->scratchpad_registry());
        }

        status_t ip_desc_create(inner_product_desc_t *ipd) {
            const bool to_ip = true;

            // reinterpret dst without spatial
            memory_desc_t ip_dst_d;
            CHECK(reshape_dst(&ip_dst_d, &dst_md_));

            // reinterpret weights without groups
            memory_desc_t ip_weights_d;
            CHECK(maybe_reshape_weights(
                    &ip_weights_d, &weights_md_, with_groups(), to_ip));

            return ip_desc_init(ipd, desc()->prop_kind, &src_md_, &ip_weights_d,
                    &bias_md_, &ip_dst_d);
        }

        status_t check_tag(const memory_desc_t &md, const format_tag_t tag) {
            const memory_desc_wrapper mdw(&md);
            if (mdw.matches_one_of_tag(tag) == format_tag::undef)
                return status::unimplemented;
            return status::success;
        }

        status_t set_and_or_check_formats(bool is_set_allowed) {
            using namespace format_tag;
            // NOTE: Only plain layouts should be supported since the dims of
            // dst_md_ must be reshaped from {N, C, H, W} to {N, C}. If the
            // conv layout is blocked by channel, then the ip layout will also
            // be blocked by channel (eg nChw16c -> nC16c). This can lead to
            // deployment of reference ip as well as strange weights layouts.
            auto atag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            if (is_set_allowed && src_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(src_md_, atag));
            else
                CHECK(check_tag(src_md_, atag));
            if (is_set_allowed && dst_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(dst_md_, atag));
            else
                CHECK(check_tag(dst_md_, atag));
            if (bias_md_.format_kind != format_kind::undef) {
                auto btag = x;
                if (bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(bias_md_, btag));
                else
                    CHECK(check_tag(bias_md_, btag));
            }
            return status::success;
        }
    };

    ip_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        CHECK(pd()->ip_pd_->create_primitive(ip_p_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> ip_p_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
