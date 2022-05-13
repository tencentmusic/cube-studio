/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_I8I8_BINARY_HPP
#define CPU_X64_JIT_UNI_I8I8_BINARY_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu/cpu_binary_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct i8i8_binary_kernel_t;

template <data_type_t src0_type, data_type_t src1_type, data_type_t dst_type>
struct jit_uni_i8i8_binary_t : public primitive_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("jit:uni:i8i8", jit_uni_i8i8_binary_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const bool ok = src_md(0)->data_type == src0_type
                    && src_md(1)->data_type == src1_type
                    && dst_md(0)->data_type == dst_type
                    && set_default_params()
                            == status::success /* should precede comparison */
                    && !has_zero_dim_memory() && is_applicable()
                    && attr()->has_default_values(sm::post_ops | sm::scales)
                    && post_ops_ok(attr(), src_md(0))
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask());
            if (!ok) return status::unimplemented;

            return status::success;
        };

    private:
        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }

        bool is_bcast_pattern(const dims_t &bcast_dims, const dim_t ndims,
                const dim_t N_bcast, const dim_t C_bcast,
                const dim_t W_bcast) const {
            return bcast_dims[0] == N_bcast && bcast_dims[1] == C_bcast
                    && bcast_dims[ndims - 1] == W_bcast;
        }

        bool is_bcast_pattern(const dims_t &bcast_dims, const dim_t N_bcast,
                const dim_t C_bcast) const {
            return bcast_dims[0] == N_bcast && bcast_dims[1] == C_bcast;
        }

        bool is_bcast_allowed(const int ndims, const dims_t &bcast_dims) const {
            // supported cases: NxCxDxHxW:{NxCx1x1x1,1xCx1x1x1,Nx1x1x1xW,
            //                            1x1x1x1xW,1x1x1x1x1}
            bool ok = true;
            // check that SP (without W) dimensions are broadcasted
            for (int d = 2; d < ndims - 1; ++d)
                ok = ok && bcast_dims[d] == 1;
            if (ndims > 2)
                ok = ok
                        && (is_bcast_pattern(bcast_dims, ndims, 0, 0, 1)
                                || is_bcast_pattern(bcast_dims, ndims, 1, 0, 1)
                                || is_bcast_pattern(bcast_dims, ndims, 0, 1, 0)
                                || is_bcast_pattern(bcast_dims, ndims, 1, 1, 0)
                                || is_bcast_pattern(
                                        bcast_dims, ndims, 1, 1, 1));
            else
                ok = ok
                        && (is_bcast_pattern(bcast_dims, 0, 0)
                                || is_bcast_pattern(bcast_dims, 1, 0)
                                || is_bcast_pattern(bcast_dims, 1, 1));
            return ok;
        }

        bool is_applicable() {
            const memory_desc_wrapper src0_d(src_md(0));
            const memory_desc_wrapper src1_d(src_md(1));
            const memory_desc_wrapper dst_d(dst_md());

            // check density first to avoid same non-dense src0 and src1 to pass
            // the next check
            bool ok = src0_d.is_dense(true) && src1_d.is_dense(true)
                    && dst_d.is_dense(true);
            if (!ok) return false;

            const auto ndims = src0_d.ndims();
            const dim_t C = ndims >= 2 ? src0_d.dims()[1] : 1;
            const bool has_oc_tail = C != src0_d.padded_dims()[1];

            // Disable compare operations when blocked tag with tail.
            // Tail processing is not supported and the vcmps instruction
            // overwrites the output vector.
            if (utils::one_of(desc()->alg_kind, alg_kind::binary_ge,
                        alg_kind::binary_gt, alg_kind::binary_le,
                        alg_kind::binary_lt, alg_kind::binary_eq,
                        alg_kind::binary_ne)
                    && has_oc_tail)
                return false;

            // full tensor operation
            if (src0_d.similar_to(src1_d, true, false, 0)) return true;
            // source0 broadcast not supported
            if (!src0_d.similar_to(dst_d, true, false, 0)) return false;

            // broadcast operation
            const auto &bcast_dims = broadcast_dims();
            if (ndims < 2 || !is_bcast_allowed(ndims, bcast_dims)) return false;

            const auto &bd0 = src0_d.blocking_desc();
            const auto &bd1 = src1_d.blocking_desc();
            // disable blocked tag for source1 when W is not broadcast
            return bd0.strides[1] == 1 && bd0.inner_nblks == 0
                    && IMPLICATION(
                            bcast_dims[ndims - 1] == 0, bd1.inner_nblks == 0);
        }
    };

    jit_uni_i8i8_binary_t(const pd_t *apd);
    ~jit_uni_i8i8_binary_t();

    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    static bool post_ops_ok(
            const primitive_attr_t *attr, const memory_desc_wrapper &d);
    std::unique_ptr<i8i8_binary_kernel_t> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
