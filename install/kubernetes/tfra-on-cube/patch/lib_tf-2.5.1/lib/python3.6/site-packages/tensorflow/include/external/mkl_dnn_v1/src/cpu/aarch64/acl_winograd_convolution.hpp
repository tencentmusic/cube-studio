/*******************************************************************************
* Copyright 2020 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_WINOGRAD_CONVOLUTION_HPP
#define CPU_AARCH64_ACL_WINOGRAD_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/aarch64/acl_convolution_utils.hpp"

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct acl_wino_resource_t : public resource_t {
    acl_wino_resource_t()
        : acl_wino_obj_(utils::make_unique<
                acl_obj_t<arm_compute::NEWinogradConvolutionLayer>>()) {}

    status_t configure(const acl_conv_conf_t &acp) {
        if (!acl_wino_obj_) return status::out_of_memory;

        // Init Compute Library tensors based on info from descriptor
        acl_wino_obj_->src_tensor.allocator()->init(acp.src_info);
        acl_wino_obj_->wei_tensor.allocator()->init(acp.wei_info);
        acl_wino_obj_->dst_tensor.allocator()->init(acp.dst_info);
        acl_wino_obj_->bia_tensor.allocator()->init(acp.bia_info);

        // clang-format off
        acl_wino_obj_->conv.configure(
            &acl_wino_obj_->src_tensor,
            &acl_wino_obj_->wei_tensor,
            acp.with_bias ? &acl_wino_obj_->bia_tensor : nullptr,
            &acl_wino_obj_->dst_tensor,
            acp.padstride_info,
            acp.act_info,
            true); // to support 5x5, 7x7 filter shapes in addition to 3x3
        // clang-format on

        return status::success;
    }

    acl_obj_t<arm_compute::NEWinogradConvolutionLayer> &get_acl_obj() const {
        return *acl_wino_obj_;
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(acl_wino_resource_t);

private:
    std::unique_ptr<acl_obj_t<arm_compute::NEWinogradConvolutionLayer>>
            acl_wino_obj_;
}; // acl_wino_resource_t

struct acl_wino_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), acp_() {}

        DECLARE_COMMON_PD_T(
                "wino:acl", acl_wino_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && utils::one_of(desc()->alg_kind,
                            alg_kind::convolution_auto,
                            alg_kind::convolution_winograd)
                    && expect_data_types(data_type::f32, data_type::f32,
                            data_type::f32, data_type::f32, data_type::f32)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops,
                            data_type::f32)
                    && !has_zero_dim_memory() && post_ops_ok();
            if (!ok) return status::unimplemented;

            auto conf_status = acl_convolution_utils::init_conf_wino(acp_,
                    src_md_, weights_md_, dst_md_, bias_md_, *desc(), *attr());
            if (conf_status != status::success) return status::unimplemented;

            set_default_alg_kind(alg_kind::convolution_winograd);

            // Number of threads in Compute Library is set by OMP_NUM_THREADS
            // dnnl_get_max_threads() == OMP_NUM_THREADS
            arm_compute::Scheduler::get().set_num_threads(
                    dnnl_get_max_threads());

            // TODO: remove dependence on scratchpad memory
            // Using user provided memory for the biases currently segfaults
            if (acp_.with_bias) {
                auto scratchpad = scratchpad_registry().registrar();
                const size_t bia_mem_sz_ = acp_.bia_info.tensor_shape()[0];
                scratchpad.template book<data_t>(
                        memory_tracking::names::key_none, bia_mem_sz_);
            }

            return status::success;
        }

        acl_conv_conf_t acp_;

    protected:
        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            auto is_eltwise
                    = [&](int idx) { return po.entry_[idx].is_eltwise(); };

            bool eltwise_ok = false;
            // Compute Library supports only one eltwise post-op
            if (po.len() == 1 && is_eltwise(0)) {
                const auto act_type = po.entry_[0].eltwise.alg;
                eltwise_ok = acl_convolution_utils::acl_act_ok(act_type);
            }

            return eltwise_ok || (po.len() == 0);
        }
    };

    acl_wino_convolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;

        auto r = utils::make_unique<acl_wino_resource_t>();
        if (!r) return status::out_of_memory;

        // Configure the resource based on information from primitive descriptor
        auto st = r->configure(pd()->acp_);
        if (st == status::success) { mapper.add(this, std::move(r)); }

        return st;
    }

    ~acl_wino_convolution_fwd_t() {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

}; // acl_wino_convolution_fwd_t

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_WINOGRAD_CONVOLUTION_HPP
