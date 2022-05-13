/*******************************************************************************
* Copyright 2018-2021 Intel Corporation
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

#ifndef CPU_RNN_REF_RNN_HPP
#define CPU_RNN_REF_RNN_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm/os_blas.hpp"

#include "cpu/rnn/cpu_rnn_pd.hpp"
#include "cpu/rnn/postgemm_dispatcher.hpp"
#include "cpu/rnn/rnn_utils.hpp"
#if DNNL_X64
#include "cpu/x64/brgemm/brgemm.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
template <typename gates_t, typename acc_t>
// The loop body needs to be put in a function as some versions of icc have
// an issue with lambdas & macros inside omp simd loops
inline void body_loop(int i, int k, const gates_t *ws_gates, acc_t *diff_bias,
        const rnn_utils::rnn_conf_t &rnn) {
    for (int j = 0; j < rnn.mb; j++)
        diff_bias[i * rnn.dhc + k]
                += ws_gates[j * rnn.scratch_gates_ld + i * rnn.dhc + k];
}
} // namespace

template <typename gates_t, typename acc_t>
void gates_reduction(const rnn_utils::rnn_conf_t &rnn, const gates_t *ws_gates_,
        acc_t *diff_bias_) {

    // @todo block k on simd-width to enable vectorization in
    // parallel_nd path
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_OMP && _OPENMP >= 201307 \
        && __INTEL_COMPILER != 1910
#pragma omp parallel for simd collapse(2)
    for (int i = 0; i < rnn.n_gates; i++)
        for (int k = 0; k < rnn.dhc; k++)
            body_loop(i, k, ws_gates_, diff_bias_, rnn);
#else
    parallel_nd(rnn.n_gates, rnn.dhc,
            [&](int i, int k) { body_loop(i, k, ws_gates_, diff_bias_, rnn); });
#endif
}

template <prop_kind_t aprop, impl::data_type_t src_type,
        impl::data_type_t weights_type, impl::data_type_t acc_type>
struct _ref_rnn_common_t : public primitive_t {
    static constexpr impl::data_type_t scratch_type
            = aprop == prop_kind::forward ? acc_type : src_type;

    /* These types are defined for each element in the cell execution */
    typedef typename prec_traits<src_type>::type src_layer_t;
    typedef typename prec_traits<src_type>::type src_iter_t;
    typedef typename prec_traits<src_type>::type dst_layer_t;
    typedef typename prec_traits<src_type>::type dst_iter_t;
    typedef typename prec_traits<weights_type>::type weights_t;
    typedef typename prec_traits<src_type>::type gemm_data_t;
    typedef typename prec_traits<acc_type>::type gemm_acc_t;
    typedef typename prec_traits<scratch_type>::type scratch_t;
    typedef typename prec_traits<src_type>::type ht_t;
    typedef typename prec_traits<src_type>::type gates_t;

    using class_name
            = _ref_rnn_common_t<aprop, src_type, weights_type, acc_type>;

    typedef rnn_cell_execution_sig((class_name::*cell_execution_f));
    typedef rnn_grid_execution_sig((class_name::*grid_execution_f));

    typedef rnn_gemm_sig((class_name::*gemm_t));
    typedef rnn_bias_prepare_sig((class_name::*bias_prepare_t));
    typedef rnn_bias_finalize_sig((class_name::*bias_finalize_t));
    typedef rnn_weights_assign_sig((class_name::*weights_assign_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    cpu_rnn_fwd_pd_t, cpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {
        using base_pd_t::base_pd_t;

        DECLARE_COMMON_PD_T("ref:any", class_name, USE_GLOBAL_SCRATCHPAD);

        status_t init_ref(engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            using namespace format_tag;
            using namespace rnn_utils;
            const alg_kind_t cell_kind = this->desc()->cell_kind;

            data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;

            bool ok = true
                    && one_of(cell_kind, alg_kind::vanilla_rnn,
                            alg_kind::vanilla_lstm, alg_kind::vanilla_gru,
                            alg_kind::lbr_gru)
                    && IMPLICATION(aprop == prop_kind::forward,
                            one_of(this->desc()->prop_kind, forward_training,
                                    forward_inference))
                    && IMPLICATION(aprop == backward,
                            one_of(this->desc()->prop_kind, backward))
                    && src_layer_dt == src_type
                    && everyone_is(
                            weights_type, weights_iter_dt, weights_layer_dt)
                    && this->set_default_params() == status::success
                    && this->with_bias();
            if (!ok) return status::unimplemented;

            rnn_.is_brgemm = false;
            ok = init_conf<class_name>(rnn_, *this->desc(), this->src_md(0),
                    this->src_md(1), this->src_md(2), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0),
                    this->dst_md(1), this->dst_md(2));
            if (!ok) return status::unimplemented;

            /* check that only supported attr have been passed */
            primitive_attr_t::skip_mask_t attr_mask
                    = primitive_attr_t::skip_mask_t::rnn_tparams;
            if (weights_layer_dt == data_type::s8)
                attr_mask = attr_mask
                        | primitive_attr_t::skip_mask_t::rnn_data_qparams
                        | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                        | primitive_attr_t::skip_mask_t::
                                rnn_weights_projection_qparams;
            ok = ok && this->attr()->has_default_values(attr_mask);
            if (!ok) return status::unimplemented;

            // Set weights descriptors to desired format
            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_, new_weights_layer_md,
                    rnn_utils::weights_type_t::layer));
            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_layer_md_ != new_weights_layer_md)
                    return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_, new_weights_iter_md,
                    rnn_utils::weights_type_t::iter));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_.format_kind
                    == format_kind::rnn_packed) {
                if (this->weights_iter_md_ != new_weights_iter_md)
                    return status::unimplemented;
            }

            if (rnn_.is_lstm_projection) {
                memory_desc_t new_weights_projection_md
                        = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
                CHECK(set_expected_desc(rnn_, new_weights_projection_md,
                        rnn_utils::weights_type_t::projection));
                if (this->weights_projection_md_.format_kind
                        == format_kind::any) {
                    this->weights_projection_md_ = new_weights_projection_md;
                } else if (this->weights_projection_md_.format_kind
                        == format_kind::rnn_packed) {
                    if (this->weights_projection_md_
                            != new_weights_projection_md)
                        return status::unimplemented;
                }
            }

            CHECK(this->check_layout_consistency());

            set_conf<class_name>(rnn_, *this->desc(), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                    this->diff_weights_md(0), this->diff_weights_md(1),
                    this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));
            return status::success;
        }

        status_t configure_brgemm() {
#if DNNL_X64
            rnn_.M = rnn_.mb;
            rnn_.N = rnn_.dhc;
            rnn_.K1 = rnn_.slc;
            rnn_.K2 = rnn_.sic;

            rnn_.nthr = dnnl_get_max_threads();

            int padding = (rnn_.is_int8()) ? 4 : (rnn_.is_bf16()) ? 2 : 1;
            rnn_.K1padded = utils::rnd_up(rnn_.K1, padding);
            rnn_.K2padded = utils::rnd_up(rnn_.K2, padding);
            if ((rnn_.is_int8() && x64::mayiuse(x64::avx512_core_bf16_amx_int8))
                    || (rnn_.is_bf16()
                            && x64::mayiuse(x64::avx512_core_bf16_amx_bf16))) {
                const dim_t max_row_width
                        = (rnn_.is_int8()
                                  && x64::mayiuse(
                                          x64::avx512_core_bf16_amx_int8))
                        ? 64
                        : 32;
                rnn_.k1_block = nstl::min(rnn_.K1, (dim_t)max_row_width);
                rnn_.k2_block = nstl::min(rnn_.K2, (dim_t)max_row_width);
                if (rnn_.k1_block <= rnn_.K1 || rnn_.k2_block <= rnn_.K2) {
                    dim_t t_k_block = nstl::min(rnn_.k1_block, rnn_.k2_block);
                    rnn_.k2_block = rnn_.k1_block = t_k_block;
                }
                rnn_.KB1_blocks = rnn_.K1 / rnn_.k1_block;
                rnn_.KB2_blocks = rnn_.K2 / rnn_.k2_block;
                rnn_.k1_tail = rnn_.K1 % rnn_.k1_block;
                rnn_.k2_tail = rnn_.K2 % rnn_.k2_block;

                if ((rnn_.k1_tail % padding || rnn_.k2_tail % padding)
                        || (rnn_.k1_block % padding
                                || rnn_.k2_block % padding)) {
                    rnn_.k1_block = rnn_.K1;
                    rnn_.k2_block = rnn_.K2;
                    rnn_.k1_tail = rnn_.k2_tail = 0;
                    rnn_.brgemm_isa = rnn_.is_int8() ? x64::avx512_core_vnni
                                                     : x64::avx512_core_bf16;
                } else {
                    rnn_.brgemm_isa = rnn_.is_int8()
                            ? x64::avx512_core_bf16_amx_int8
                            : x64::avx512_core_bf16_amx_bf16;
                }
            } else {
                rnn_.k1_block = rnn_.K1;
                rnn_.k2_block = rnn_.K2;
                rnn_.brgemm_isa = x64::isa_any;
            }

            rnn_.n_block = 32;
            rnn_.N_blocks = utils::div_up(rnn_.N, rnn_.n_block);
            rnn_.n_tail = rnn_.N % rnn_.n_block;

            float work_by_N = (float)rnn_.N_blocks / (float)rnn_.nthr;
            dim_t l2_cache_size = platform::get_per_core_cache_size(2);

            dim_t As = sizeof(src_layer_t) * rnn_.M
                    * (nstl::max(rnn_.K1, rnn_.K2));
            dim_t Cs = sizeof(scratch_t) * 5 * (rnn_.M * rnn_.n_block);

            const bool adj_by_l2 = rnn_.is_f32()
                    ? true
                    : ((float)(As + Cs) < 0.6 * (float)l2_cache_size);
            if (work_by_N > 2.0) {
                rnn_.m_block = rnn_.M;
            } else if (work_by_N > 1.0 && adj_by_l2) {
                rnn_.m_block = rnn_.M;
            } else {
                dim_t max_m_blocks
                        = ((rnn_.is_int8_amx() || rnn_.is_bf16_amx()) ? 1 : 4)
                        * utils::div_up(rnn_.nthr, rnn_.N_blocks);
                dim_t max_m_value
                        = (rnn_.is_int8_amx() || rnn_.is_bf16_amx()) ? 64 : 24;
                dim_t max_M = nstl::min(max_m_value,
                        nstl::max((dim_t)1, rnn_.M / max_m_blocks));
                dim_t min_M = 4;

                rnn_.m_block = 1;
                for (dim_t m = max_M; m >= min_M; m--)
                    if (rnn_.M % m == 0) {
                        rnn_.m_block = m;
                        break;
                    }
                if (rnn_.m_block == 1) rnn_.m_block = rnn_.M;
            }
            rnn_.M_blocks = rnn_.M / rnn_.m_block;

            rnn_.unfused_post_gemm = (rnn_.M_blocks == 1);

            rnn_.LDA1[0] = rnn_.src_layer_ld_;
            rnn_.LDA1[1] = rnn_.dst_iter_ld_;
            rnn_.LDA1[2] = rnn_.ws_states_layer_ld;

            rnn_.LDA2[0] = rnn_.src_iter_ld_;
            rnn_.LDA2[1] = rnn_.dst_layer_ld_;
            rnn_.LDA2[2] = rnn_.ws_states_iter_ld;

            rnn_.LDB1 = rnn_.n_block;
            rnn_.LDB2 = rnn_.n_block;
            rnn_.LDC = rnn_.scratch_gates_ld;

            auto get_dim = [&](dim_t block, dim_t tail) {
                return (block == 0) ? tail : block;
            };
            dim_t n_block = nstl::min(rnn_.N, rnn_.n_block);
            dim_t n_tail = nstl::min(rnn_.N, rnn_.nproj_tail);
            if (rnn_.LDA1[0] < rnn_.k1_block && rnn_.LDA1[1] < rnn_.k1_block
                    && rnn_.LDA1[2] < rnn_.k1_block)
                return status::unimplemented;
            if (rnn_.LDA2[0] < rnn_.k2_block && rnn_.LDA2[1] < rnn_.k2_block
                    && rnn_.LDA2[2] < rnn_.k2_block)
                return status::unimplemented;
            if (rnn_.LDB1 < get_dim(n_block, n_tail)
                    && rnn_.LDB2 < get_dim(n_block, n_tail))
                return status::unimplemented;
            if (rnn_.LDC < get_dim(n_block, n_tail))
                return status::unimplemented;

            rnn_.KBproj_blocks = 0;
            if (rnn_.is_lstm_projection) {
                rnn_.Nproj = rnn_.dic;
                rnn_.Nproj_blocks = utils::div_up(rnn_.Nproj, rnn_.n_block);
                rnn_.nproj_tail = rnn_.Nproj % rnn_.n_block;

                rnn_.Kproj = rnn_.dhc;
                rnn_.Kprojpadded = utils::rnd_up(rnn_.Kproj, padding);
                if (rnn_.is_int8_amx() || rnn_.is_bf16_amx()) {
                    const dim_t max_row_width = rnn_.is_int8_amx() ? 64 : 32;
                    rnn_.kproj_block
                            = nstl::min(rnn_.Kproj, (dim_t)max_row_width);

                    rnn_.KBproj_blocks = rnn_.Kproj / rnn_.kproj_block;
                    rnn_.kproj_tail = rnn_.Kproj % rnn_.kproj_block;

                    if ((rnn_.kproj_tail % padding)
                            || (rnn_.kproj_block % padding)) {
                        rnn_.kproj_block = rnn_.Kproj;
                        rnn_.kproj_tail = 0;
                        rnn_.brgemm_isa = rnn_.is_int8()
                                ? x64::avx512_core_vnni
                                : x64::avx512_core_bf16;
                    } else {
                        rnn_.brgemm_isa = rnn_.is_int8()
                                ? x64::avx512_core_bf16_amx_int8
                                : x64::avx512_core_bf16_amx_bf16;
                    }
                } else {
                    rnn_.kproj_block = rnn_.Kproj;
                    rnn_.KBproj_blocks = rnn_.Kproj / rnn_.kproj_block;
                }
                rnn_.LDAproj = rnn_.proj_ht_ld;
                rnn_.LDBproj = rnn_.n_block;
                if (rnn_.dt_conf != rnn_utils::all_f32) {
                    rnn_.LDCproj[0] = rnn_.scratch_gates_ld;
                } else {
                    rnn_.LDCproj[0] = rnn_.scratch_ht_ld;
                    rnn_.LDCproj[1] = rnn_.dst_layer_ld_;
                    rnn_.LDCproj[2] = rnn_.dst_iter_ld_;
                    rnn_.LDCproj[3] = rnn_.ws_states_layer_ld;
                }

                dim_t n_block = nstl::min(rnn_.Nproj, rnn_.n_block);
                dim_t n_tail = nstl::min(rnn_.Nproj, rnn_.nproj_tail);
                bool check_LDC = false;
                if (rnn_.dt_conf != rnn_utils::all_f32) {
                    check_LDC = rnn_.LDCproj[0] < get_dim(n_block, n_tail);
                } else {
                    check_LDC = rnn_.LDCproj[0] < get_dim(n_block, n_tail)
                            && rnn_.LDCproj[1] < get_dim(n_block, n_tail)
                            && rnn_.LDCproj[2] < get_dim(n_block, n_tail)
                            && rnn_.LDCproj[3] < get_dim(n_block, n_tail);
                }
                if (rnn_.LDAproj < rnn_.kproj_block
                        || rnn_.LDBproj < get_dim(n_block, n_tail) || check_LDC)
                    return status::unimplemented;
            }
            return status::success;
#else
            return status::unimplemented;
#endif
        }

        status_t init_brgemm(engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            using namespace format_tag;
            using namespace rnn_utils;
#if DNNL_X64
            using namespace x64;
            const alg_kind_t cell_kind = this->desc()->cell_kind;

            data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
            data_type_t weights_iter_dt
                    = this->desc()->weights_iter_desc.data_type;
            data_type_t weights_layer_dt
                    = this->desc()->weights_layer_desc.data_type;

            if (aprop == backward || one_of(this->desc()->prop_kind, backward))
                return status::unimplemented;
            bool ok = true && one_of(cell_kind, alg_kind::vanilla_lstm)
                    && IMPLICATION(aprop == prop_kind::forward,
                            one_of(this->desc()->prop_kind, forward_inference))
                    && src_layer_dt == src_type
                    && everyone_is(
                            weights_type, weights_iter_dt, weights_layer_dt)
                    && this->set_default_params() == status::success
                    && this->with_bias();
            if (!ok) return status::unimplemented;

            rnn_.is_brgemm = true;
            ok = init_conf<class_name>(rnn_, *this->desc(), this->src_md(0),
                    this->src_md(1), this->src_md(2), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION), this->dst_md(0),
                    this->dst_md(1), this->dst_md(2));
            if (!ok) return status::unimplemented;

            if (rnn_.is_bf16() && !mayiuse(avx512_core_bf16))
                return status::unimplemented;
            if (rnn_.is_int8() && !mayiuse(avx512_core_vnni))
                return status::unimplemented;
            if (rnn_.is_f32() && !mayiuse(avx512_core))
                return status::unimplemented;

            /* check that only supported attr have been passed */
            primitive_attr_t::skip_mask_t attr_mask
                    = primitive_attr_t::skip_mask_t::rnn_tparams;
            if (weights_layer_dt == data_type::s8)
                attr_mask = attr_mask
                        | primitive_attr_t::skip_mask_t::rnn_data_qparams
                        | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                        | primitive_attr_t::skip_mask_t::
                                rnn_weights_projection_qparams;
            ok = ok && this->attr()->has_default_values(attr_mask);
            if (!ok) return status::unimplemented;

            set_conf<class_name>(rnn_, *this->desc(), this->weights_md(0),
                    this->weights_md(1),
                    this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION),
                    this->diff_weights_md(0), this->diff_weights_md(1),
                    this->arg_md(DNNL_ARG_DIFF_WEIGHTS_PROJECTION));

            CHECK(configure_brgemm());

            // Set weights descriptors to desired format
            memory_desc_t new_weights_layer_md = *this->weights_md(0);
            CHECK(set_expected_desc(rnn_, new_weights_layer_md,
                    rnn_utils::weights_type_t::layer));
            if (this->weights_layer_md_.format_kind == format_kind::any) {
                this->weights_layer_md_ = new_weights_layer_md;
            } else if (this->weights_layer_md_ != new_weights_layer_md) {
                return status::unimplemented;
            }

            memory_desc_t new_weights_iter_md = *this->weights_md(1);
            CHECK(set_expected_desc(rnn_, new_weights_iter_md,
                    rnn_utils::weights_type_t::iter));
            if (this->weights_iter_md_.format_kind == format_kind::any) {
                this->weights_iter_md_ = new_weights_iter_md;
            } else if (this->weights_iter_md_ != new_weights_iter_md) {
                return status::unimplemented;
            }
            if (rnn_.is_lstm_projection) {
                memory_desc_t new_weights_projection_md
                        = *this->arg_md(DNNL_ARG_WEIGHTS_PROJECTION);
                CHECK(set_expected_desc(rnn_, new_weights_projection_md,
                        rnn_utils::weights_type_t::projection));
                if (this->weights_projection_md_.format_kind
                        == format_kind::any) {
                    this->weights_projection_md_ = new_weights_projection_md;
                } else if (this->weights_projection_md_
                        != new_weights_projection_md) {
                    return status::unimplemented;
                }
            }
            if (rnn_.is_int8()) {
                const memory_desc_wrapper &weights_layer_d(
                        this->weights_layer_md_);
                const memory_desc_wrapper &weights_iter_d(
                        this->weights_iter_md_);
                const auto &pdims_l = weights_layer_d.padded_dims();
                const auto &pdims_i = weights_iter_d.padded_dims();
                rnn_.weights_layer_comp_offset = rnn_.n_layer * rnn_.n_dir
                        * rnn_.n_gates * pdims_l[2] * pdims_l[4];
                rnn_.weights_iter_comp_offset = rnn_.n_layer * rnn_.n_dir
                        * rnn_.n_gates * pdims_i[2] * pdims_i[4];
                if (rnn_.is_lstm_projection) {
                    const memory_desc_wrapper &weights_proj_d(
                            this->weights_projection_md_);
                    const auto &pdims_p = weights_proj_d.padded_dims();
                    rnn_.weights_projection_comp_offset = rnn_.n_layer
                            * rnn_.n_dir * pdims_p[2] * pdims_p[3];
                } else {
                    rnn_.weights_projection_comp_offset = 0;
                }
            }
            CHECK(this->check_layout_consistency());

            return status::success;
#else
            return status::unimplemented;
#endif
        }

        status_t init(engine_t *engine) {
            status_t st = init_brgemm(engine);
            if (st != status::success) {
                rnn_.is_brgemm = false;
                st = init_ref(engine);
            }
            if (st == status::success) {
                size_t scratchpad_sz {0}, ws_sz {0};
                get_scratchpad_and_workspace_sizes(rnn_, scratchpad_sz, ws_sz);

                init_scratchpad(scratchpad_sz);
                // initialize the workspace if needed
                if (rnn_.is_training) {
                    dims_t ws_dims = {(dim_t)ws_sz};
                    dnnl_memory_desc_init_by_tag(&this->ws_md_, 1, ws_dims,
                            data_type::u8, format_tag::x);
                }
            }
            return st;
        }

        rnn_utils::rnn_conf_t rnn_;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();

            {
                size_t data_size = 1; // "true" data size already incorporated
                size_t data_align = alignof(float); // "worst" case scenario
                size_t perf_align = 4096;
                scratchpad.book(key_rnn_space, scratchpad_sz, data_size,
                        data_align, perf_align);
            }

            int max_nparts = this->cell_kind() == alg_kind::vanilla_gru ? 2 : 1;
            int ptr_wei_sz = rnn_.n_layer * rnn_.n_dir * max_nparts;
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_layer, ptr_wei_sz);
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_iter, ptr_wei_sz);
            scratchpad.template book<float *>(
                    key_rnn_ptrs_wei_projection, ptr_wei_sz);
            scratchpad.template book<float *>(key_rnn_ptrs_bia, ptr_wei_sz);
            scratchpad.template book<scratch_t>(
                    key_rnn_gates, rnn_.scratch_gates_size);
            scratchpad.template book<ht_t>(key_rnn_ht, rnn_.scratch_ht_size);
            scratchpad.template book<gemm_acc_t>(
                    key_rnn_diff_ht, rnn_.scratch_diff_ht_size);
            scratchpad.template book<scratch_t>(
                    key_rnn_cell, rnn_.scratch_cell_size);
#if DNNL_X64
            if (rnn_.is_brgemm) {
                if (rnn_.is_int8_amx() || rnn_.is_bf16_amx()) {
                    size_t n_elements = rnn_.m_block * rnn_.n_block;
                    scratchpad.template book<gemm_acc_t>(
                            key_brgemm_primitive_buffer,
                            rnn_.nthr * n_elements);

                    int max_K_Block = nstl::max(rnn_.KB1_blocks + 1,
                            nstl::max(rnn_.KBproj_blocks + 1,
                                    rnn_.KB2_blocks + 1));
                    scratchpad.template book<x64::brgemm_batch_element_t>(
                            key_brgemm_primitive_batch,
                            max_K_Block * rnn_.nthr);
                } else {
                    scratchpad.template book<x64::brgemm_batch_element_t>(
                            key_brgemm_primitive_batch, rnn_.nthr);
                }
            }
#endif
        }
    };

    _ref_rnn_common_t(const pd_t *apd)
        : primitive_t(apd), rnn_postgemm_(nullptr) {}

    status_t init(engine_t *engine) override {
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dhc and sic != dhc
        /// respectively

        bias_preparation_func = &class_name::bias_prepare;
        bias_finalization_func = &class_name::bias_finalize;

        auto set_gemm_funcs = [](bool packed_gemm, gemm_t &g,
                                      weights_assign_t &a, bool is_brgemm) {
            if (packed_gemm) {
                g = &class_name::packed_gemm;
                a = &class_name::assign_packed_weights;
            } else {
                g = (!is_brgemm) ? &class_name::gemm : nullptr;
                a = &class_name::assign_weights;
            }
        };
        set_gemm_funcs(pd()->rnn_.use_iter_packed_gemm, gemm_iter_func,
                weights_iter_assign_func, pd()->rnn_.is_brgemm);

        set_gemm_funcs(pd()->rnn_.use_layer_packed_gemm, gemm_layer_func,
                weights_layer_assign_func, pd()->rnn_.is_brgemm);

        if (pd()->rnn_.is_lstm_projection) {
            set_gemm_funcs(pd()->rnn_.use_projection_packed_gemm,
                    gemm_projection_func, weights_projection_assign_func,
                    pd()->rnn_.is_brgemm);
        }

        rnn_postgemm_ = new rnn_postgemm_dispatcher<aprop, src_type,
                scratch_type, acc_type>(pd()->rnn_, pd());
        assert(rnn_postgemm_ != nullptr);
        switch (pd()->cell_kind()) {
            case alg_kind::vanilla_rnn:
            case alg_kind::vanilla_lstm:
                cell_func = (pd()->rnn_.is_brgemm)
                        ? &class_name::cell_execution_brgemm
                        : &class_name::cell_execution_ref;
                break;
            case alg_kind::vanilla_gru:
                cell_func = &class_name::cell_execution_gru;
                break;
            case alg_kind::lbr_gru:
                cell_func = &class_name::cell_execution_gru_lbr;
                break;
            default: break;
        }

        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_, ws_gates_offset_, ws_ht_offset_,
                ws_states_layer_offset_, ws_states_iter_offset_,
                ws_states_iter_c_offset_, ws_diff_states_layer_offset_,
                ws_diff_states_iter_offset_, ws_diff_states_iter_c_offset_,
                ws_grid_comp_offset_, ws_bias_offset_, scratch_gates_offset_,
                scratch_ht_offset_, scratch_diff_ht_offset_,
                scratch_cell_offset_, scratchpad_size, workspace_size);
#if DNNL_X64
        auto rnn = pd()->rnn_;

        auto init_brgemm = [&](x64::brgemm_t *desc, x64::cpu_isa_t isa,
                                   std::unique_ptr<x64::brgemm_kernel_t> &ker,
                                   dim_t M, dim_t N, dim_t K, dim_t LDA,
                                   dim_t LDB, dim_t LDC, float beta) {
            bool transA = false;
            bool transB = false;
            x64::brgemm_layout_t layout = x64::brgemm_row_major;
            CHECK(brgemm_desc_init(desc, isa, x64::brgemm_addr, src_type,
                    weights_type, transA, transB, layout, 1.0, beta, LDA, LDB,
                    LDC, M, N, K));

            if (!rnn.is_int8_amx() && !rnn.is_bf16_amx()) {
                x64::brgemm_attr_t brgattr;
                brgattr.max_bs = 1;
                brgattr.max_top_vpad = 0;
                brgattr.max_bottom_vpad = 0;
                CHECK(brgemm_desc_set_attr(desc, brgattr));
            }

            x64::brgemm_kernel_t *_t_ptr;
            CHECK(brgemm_kernel_create(&_t_ptr, *desc));
            CHECK(safe_ptr_assign<x64::brgemm_kernel_t>(ker, _t_ptr));
            return status::success;
        };

        if (pd()->rnn_.is_brgemm) {
            int brgemm_n = nstl::min(rnn.N, rnn.n_block);
            int brgemm_n_tail = nstl::min(rnn.N, rnn.n_tail);
            for (int i = 0; i < 3; i++) {
                init_brgemm(&brgemm_desc_layer_b0_[i], rnn.brgemm_isa,
                        brgemm_kernel_layer_b0_[i], rnn.m_block, brgemm_n,
                        rnn.k1_block, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0);
                init_brgemm(&brgemm_desc_iter_b0_[i], rnn.brgemm_isa,
                        brgemm_kernel_iter_b0_[i], rnn.m_block, brgemm_n,
                        rnn.k2_block, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 0.0);
                init_brgemm(&brgemm_desc_iter_b1_[i], rnn.brgemm_isa,
                        brgemm_kernel_iter_b1_[i], rnn.m_block, brgemm_n,
                        rnn.k2_block, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0);
                if (rnn.n_tail) {
                    init_brgemm(&brgemm_desc_layer_N_tail_b0_[i],
                            rnn.brgemm_isa, brgemm_kernel_layer_N_tail_b0_[i],
                            rnn.m_block, brgemm_n_tail, rnn.k1_block,
                            rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0);
                    init_brgemm(&brgemm_desc_iter_N_tail_b0_[i], rnn.brgemm_isa,
                            brgemm_kernel_iter_N_tail_b0_[i], rnn.m_block,
                            brgemm_n_tail, rnn.k2_block, rnn.LDA2[i], rnn.LDB2,
                            rnn.LDC, 0.0);
                    init_brgemm(&brgemm_desc_iter_N_tail_b1_[i], rnn.brgemm_isa,
                            brgemm_kernel_iter_N_tail_b1_[i], rnn.m_block,
                            brgemm_n_tail, rnn.k2_block, rnn.LDA2[i], rnn.LDB2,
                            rnn.LDC, 1.0);
                }
                if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
                    if (rnn.k1_tail)
                        init_brgemm(&brgemm_desc_layer_K1_tail_b1_[i],
                                rnn.brgemm_isa,
                                brgemm_kernel_layer_K1_tail_b1_[i], rnn.m_block,
                                brgemm_n, rnn.k1_tail, rnn.LDA1[i], rnn.LDB1,
                                rnn.LDC, 1.0);
                    if (rnn.k1_tail && rnn.n_tail)
                        init_brgemm(&brgemm_desc_layer_NK1_tail_b1_[i],
                                rnn.brgemm_isa,
                                brgemm_kernel_layer_NK1_tail_b1_[i],
                                rnn.m_block, brgemm_n_tail, rnn.k1_tail,
                                rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0);
                    if (rnn.k2_tail)
                        init_brgemm(&brgemm_desc_iter_K2_tail_b1_[i],
                                rnn.brgemm_isa,
                                brgemm_kernel_iter_K2_tail_b1_[i], rnn.m_block,
                                brgemm_n, rnn.k2_tail, rnn.LDA2[i], rnn.LDB2,
                                rnn.LDC, 1.0);
                    if (rnn.k2_tail && rnn.n_tail)
                        init_brgemm(&brgemm_desc_iter_NK2_tail_b1_[i],
                                rnn.brgemm_isa,
                                brgemm_kernel_iter_NK2_tail_b1_[i], rnn.m_block,
                                brgemm_n_tail, rnn.k2_tail, rnn.LDA2[i],
                                rnn.LDB2, rnn.LDC, 1.0);
                }
            }
            if (rnn.is_lstm_projection) {
                dim_t brgemm_np = nstl::min(rnn.Nproj, rnn.n_block);
                dim_t brgemm_np_tail = nstl::min(rnn.Nproj, rnn.nproj_tail);
                int n_kernel = (rnn.dt_conf == rnn_utils::all_f32) ? 4 : 1;
                for (int i = 0; i < n_kernel; i++) {
                    init_brgemm(&brgemm_desc_proj_b0_[i], rnn.brgemm_isa,
                            brgemm_kernel_proj_b0_[i], rnn.m_block, brgemm_np,
                            rnn.kproj_block, rnn.LDAproj, rnn.LDBproj,
                            rnn.LDCproj[i], 0.0);
                    if (rnn.nproj_tail) {
                        init_brgemm(&brgemm_desc_proj_N_tail_b0_[i],
                                rnn.brgemm_isa,
                                brgemm_kernel_proj_N_tail_b0_[i], rnn.m_block,
                                brgemm_np_tail, rnn.kproj_block, rnn.LDAproj,
                                rnn.LDBproj, rnn.LDCproj[i], 0.0);
                        init_brgemm(&brgemm_desc_proj_N_tail_b1_[i],
                                rnn.brgemm_isa,
                                brgemm_kernel_proj_N_tail_b1_[i], rnn.m_block,
                                brgemm_np_tail, rnn.kproj_block, rnn.LDAproj,
                                rnn.LDBproj, rnn.LDCproj[i], 1.0);
                    }
                    if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
                        if (rnn.kproj_tail)
                            init_brgemm(&brgemm_desc_proj_K_tail_b1_[i],
                                    rnn.brgemm_isa,
                                    brgemm_kernel_proj_K_tail_b1_[i],
                                    rnn.m_block, brgemm_np, rnn.kproj_tail,
                                    rnn.LDAproj, rnn.LDBproj, rnn.LDCproj[i],
                                    1.0);
                        if (rnn.kproj_tail && rnn.nproj_tail)
                            init_brgemm(&brgemm_desc_proj_NK_tail_b1_[i],
                                    rnn.brgemm_isa,
                                    brgemm_kernel_proj_NK_tail_b1_[i],
                                    rnn.m_block, brgemm_np_tail, rnn.kproj_tail,
                                    rnn.LDAproj, rnn.LDBproj, rnn.LDCproj[i],
                                    1.0);
                    }
                }
            }
            if (rnn.is_int8_amx() || rnn.is_bf16_amx()) {
                brgemm_init_tiles(brgemm_desc_layer_b0_[0], pallete_buff_);
                if (rnn.n_tail)
                    brgemm_init_tiles(brgemm_desc_layer_N_tail_b0_[0],
                            pallete_buff_n_tail_);
                if (rnn.k1_tail)
                    brgemm_init_tiles(brgemm_desc_layer_K1_tail_b1_[0],
                            pallete_buff_k1_tail_);
                if (rnn.k2_tail)
                    brgemm_init_tiles(brgemm_desc_iter_K2_tail_b1_[0],
                            pallete_buff_k2_tail_);
                if (rnn.k1_tail && rnn.n_tail)
                    brgemm_init_tiles(brgemm_desc_layer_NK1_tail_b1_[0],
                            pallete_buff_nk1_tail_);
                if (rnn.k2_tail && rnn.n_tail)
                    brgemm_init_tiles(brgemm_desc_iter_NK2_tail_b1_[0],
                            pallete_buff_nk2_tail_);
                if (rnn.is_lstm_projection) {
                    brgemm_init_tiles(
                            brgemm_desc_proj_b0_[0], pallete_buff_proj_);
                    if (rnn.nproj_tail)
                        brgemm_init_tiles(brgemm_desc_proj_N_tail_b0_[0],
                                pallete_buff_nproj_tail_);
                    if (rnn.kproj_tail)
                        brgemm_init_tiles(brgemm_desc_proj_K_tail_b1_[0],
                                pallete_buff_kproj_tail_);
                    if (rnn.kproj_tail && rnn.nproj_tail)
                        brgemm_init_tiles(brgemm_desc_proj_NK_tail_b1_[0],
                                pallete_buff_nkproj_tail_);
                }
            }
        }
#endif
        return status::success;
    }

    ~_ref_rnn_common_t() { delete rnn_postgemm_; }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_(ctx);
        return status::success;
    }

private:
#if DNNL_X64
    x64::brgemm_t brgemm_desc_layer_b0_[3];
    x64::brgemm_t brgemm_desc_iter_b0_[3];
    x64::brgemm_t brgemm_desc_iter_b1_[3];
    x64::brgemm_t brgemm_desc_layer_N_tail_b0_[3];
    x64::brgemm_t brgemm_desc_iter_N_tail_b0_[3];
    x64::brgemm_t brgemm_desc_iter_N_tail_b1_[3];

    x64::brgemm_t brgemm_desc_layer_K1_tail_b1_[3];
    x64::brgemm_t brgemm_desc_layer_NK1_tail_b1_[3];
    x64::brgemm_t brgemm_desc_iter_K2_tail_b1_[3];
    x64::brgemm_t brgemm_desc_iter_NK2_tail_b1_[3];

    x64::brgemm_t brgemm_desc_proj_b0_[4];
    x64::brgemm_t brgemm_desc_proj_N_tail_b0_[4];
    x64::brgemm_t brgemm_desc_proj_N_tail_b1_[4];
    x64::brgemm_t brgemm_desc_proj_K_tail_b1_[4];
    x64::brgemm_t brgemm_desc_proj_NK_tail_b1_[4];

    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_layer_b0_[3];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_iter_b0_[3];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_iter_b1_[3];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_layer_N_tail_b0_[3];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_iter_N_tail_b0_[3];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_iter_N_tail_b1_[3];

    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_layer_K1_tail_b1_[3];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_layer_NK1_tail_b1_[3];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_iter_K2_tail_b1_[3];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_iter_NK2_tail_b1_[3];

    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_proj_b0_[4];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_proj_N_tail_b0_[4];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_proj_N_tail_b1_[4];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_proj_K_tail_b1_[4];
    std::unique_ptr<x64::brgemm_kernel_t> brgemm_kernel_proj_NK_tail_b1_[4];

    char pallete_buff_[64];
    char pallete_buff_n_tail_[64];
    char pallete_buff_k1_tail_[64];
    char pallete_buff_k2_tail_[64];
    char pallete_buff_nk1_tail_[64];
    char pallete_buff_nk2_tail_[64];
    char pallete_buff_proj_[64];
    char pallete_buff_nproj_tail_[64];
    char pallete_buff_kproj_tail_[64];
    char pallete_buff_nkproj_tail_[64];
#endif
    void execute_(const exec_ctx_t &ctx) const;
    rnn_grid_execution_sig(linear_execution);
    rnn_cell_execution_sig(cell_execution_ref);
    rnn_cell_execution_sig(cell_execution_brgemm);
    rnn_cell_execution_sig(cell_execution_gru);
    rnn_cell_execution_sig(cell_execution_gru_lbr);
    rnn_gemm_sig(gemm);
    rnn_gemm_sig(packed_gemm);
    rnn_bias_prepare_sig(bias_prepare);
    rnn_bias_finalize_sig(bias_finalize);
    rnn_weights_assign_sig(assign_weights);
    rnn_weights_assign_sig(assign_packed_weights);

    float (*activation_func)(float s, float alpha, float cliping);

    void copy_init_layer(const rnn_utils::rnn_conf_t &rnn,
            src_layer_t *ws_states_layer_, gemm_acc_t *ws_diff_states_layer_,
            const src_layer_t *xt_, const gemm_acc_t *diff_dst_layer) const;

    template <typename input_t>
    void copy_init_iter(const rnn_utils::rnn_conf_t &rnn,
            src_iter_t *ws_states_iter_, float *ws_states_iter_c_,
            gemm_acc_t *ws_diff_states_iter_,
            gemm_acc_t *ws_diff_states_iter_c_, const input_t *src_iter_,
            const float *src_iter_c_, const gemm_acc_t *diff_dst_iter_,
            const float *diff_dst_iter_c_) const;

    template <typename dst_layer_dt, typename dst_iter_dt>
    void copy_res_layer(const rnn_utils::rnn_conf_t &rnn,
            dst_layer_dt *dst_layer_, gemm_acc_t *diff_src_layer_,
            const dst_iter_dt *dst_iter_, const src_layer_t *ws_states_layer_,
            const gemm_acc_t *ws_diff_states_layer_) const;

    template <typename prim_dst_iter_t, typename prim_dst_layer_t>
    void copy_res_iter(const rnn_utils::rnn_conf_t &rnn,
            prim_dst_iter_t *dst_iter_, float *dst_iter_c_,
            gemm_acc_t *diff_src_iter_, float *diff_src_iter_c_,
            const prim_dst_layer_t *dst_layer_,
            const src_iter_t *ws_states_iter_, const float *ws_states_iter_c,
            const gemm_acc_t *ws_diff_states_iter_,
            const gemm_acc_t *ws_diff_states_iter_c_) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    size_t ws_gates_offset_;
    size_t ws_ht_offset_;
    size_t ws_states_layer_offset_;
    size_t ws_states_iter_offset_;
    size_t ws_states_iter_c_offset_;
    size_t ws_bias_offset_;
    size_t ws_diff_states_layer_offset_;
    size_t ws_diff_states_iter_offset_;
    size_t ws_diff_states_iter_c_offset_;
    size_t ws_grid_comp_offset_;
    size_t scratch_gates_offset_;
    size_t scratch_ht_offset_;
    size_t scratch_diff_ht_offset_;
    size_t scratch_cell_offset_;
    rnn_postgemm_dispatcher<aprop, src_type, scratch_type, acc_type>
            *rnn_postgemm_;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    bias_prepare_t bias_preparation_func;
    bias_finalize_t bias_finalization_func;
    weights_assign_t weights_layer_assign_func;
    weights_assign_t weights_iter_assign_func;
    weights_assign_t weights_projection_assign_func;

    gemm_t gemm_layer_func;
    gemm_t gemm_iter_func;
    gemm_t gemm_projection_func;
};

using ref_rnn_fwd_f32_t = _ref_rnn_common_t<prop_kind::forward, data_type::f32,
        data_type::f32, data_type::f32>;
using ref_rnn_bwd_f32_t = _ref_rnn_common_t<prop_kind::backward, data_type::f32,
        data_type::f32, data_type::f32>;
using ref_rnn_fwd_bf16_t = _ref_rnn_common_t<prop_kind::forward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_bwd_bf16_t = _ref_rnn_common_t<prop_kind::backward,
        data_type::bf16, data_type::bf16, data_type::f32>;
using ref_rnn_fwd_u8s8_t = _ref_rnn_common_t<prop_kind::forward, data_type::u8,
        data_type::s8, data_type::s32>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
