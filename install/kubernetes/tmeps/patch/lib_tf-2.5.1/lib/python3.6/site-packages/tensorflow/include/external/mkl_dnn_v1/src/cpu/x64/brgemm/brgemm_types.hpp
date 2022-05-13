/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef CPU_X64_BRGEMM_BRGEMM_TYPES_HPP
#define CPU_X64_BRGEMM_BRGEMM_TYPES_HPP

#include "common/primitive_attr.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

// The type defines organization of batch of matrices
typedef enum {
    // A and B arrays of pointers
    brgemm_addr = 1,
    // Based address and fixed offset between matrices
    brgemm_offs = 2,
    // Base addresses and arrays of strides between matrices.
    brgemm_strd = 3,
} brgemm_batch_kind_t;

// The type defines the storage format of matrix
typedef enum {
    brgemm_col_major = 1,
    brgemm_row_major = 2,
} brgemm_layout_t;

struct brgemm_strides_t {
    // Stride between A matrices
    dim_t stride_a;
    // Stride between B matrices
    dim_t stride_b;
};

typedef enum {
    brgemm_lo_default = 0,
    brgemm_lo_bl_1load,
    brgemm_lo_bl_1bcst,
} brgemm_kernel_loop_order_t;

typedef enum {
    brgemm_prf_default = 1,
} brgemm_kernel_prefetching_t;

struct brgemm_attr_t {
    brgemm_attr_t()
        : max_bs(INT_MAX)
        , max_top_vpad(0)
        , max_bottom_vpad(0)
        , hint_expected_A_size(LLONG_MAX)
        , hint_expected_B_size(LLONG_MAX)
        , hint_expected_C_size(LLONG_MAX)
        , hint_loop_order(brgemm_kernel_loop_order_t::brgemm_lo_default)
        , hint_prefetching(brgemm_kernel_prefetching_t::brgemm_prf_default)
        , wary_tail_read(true) {}
    int max_bs;
    int max_top_vpad, max_bottom_vpad;
    dim_t hint_expected_A_size, hint_expected_B_size, hint_expected_C_size;
    brgemm_kernel_loop_order_t hint_loop_order;
    brgemm_kernel_prefetching_t hint_prefetching;
    bool wary_tail_read;
};

struct brgemm_batch_element_t {
    brgemm_batch_element_t() {
        ptr.A = ptr.B = nullptr;
        vvpad.top = vvpad.bottom = 0;
    }
    union {
        struct {
            const void *A;
            const void *B;
        } ptr;
        struct {
            dim_t A;
            dim_t B;
        } offset;
    };
    union {
        struct {
            dim_t top;
            dim_t bottom;
        } vvpad;
        struct {
            dim_t left;
            dim_t right;
        } hvpad;
    };
};

struct brgemm_t {
    int bcast_dim; // M;
    int load_dim; // N;
    int reduce_dim; // K;
    int LDA;
    int LDB;
    int LDC;
    int LDD;

    float alpha;
    float beta;

    int bdb, bd_block, bdb_tail;
    int bdb2, bd_block2, bdb2_tail;
    int ldb, ld_block, ldb_tail;
    int ldb2, ld_block2, ldb2_tail;
    int rdb, rd_block, rdb_tail;
    int rd_step, ld_step;

    impl::data_type_t dt_a;
    impl::data_type_t dt_c;
    impl::data_type_t dt_b;
    impl::data_type_t dt_d;
    impl::data_type_t dt_bias;

    int typesize_A;
    int typesize_B;
    int typesize_C;
    int typesize_D;
    int typesize_bias;

    bool is_int8, is_int8_amx;
    bool is_bf16, is_bf16_amx;
    bool is_f32;
    bool is_amx;

    dim_t stride_a; // Offset in bytes
    dim_t stride_b;

    brgemm_layout_t layout;

    brgemm_batch_kind_t type;
    bool embd_bcst;

    bool with_bias;
    bool with_sum;
    float sum_scale;
    bool with_eltwise;
    bool with_scales;
    bool req_s8s8_compensation;
    int is_oc_scale;

    const primitive_attr_t *attr;
    post_ops_t::entry_t::eltwise_t eltwise;

    brgemm_attr_t brgattr;
    static constexpr int MAX_VPAD = 100;
};

struct brgemm_kernel_params_t {
    const void *ptr_A;
    const void *ptr_B;
    const brgemm_batch_element_t *batch;
    void *ptr_C;

    const void *ptr_bias;
    void *ptr_D;

    const void *ptr_scales;
    void *ptr_buf;

    size_t do_post_ops;
    size_t BS;
};

struct jit_brgemm_kernel_base_t;

struct brgemm_kernel_t {
    brgemm_kernel_t(const brgemm_t abrd);
    ~brgemm_kernel_t();

    status_t create_kernel();
    void operator()(brgemm_kernel_params_t *) const;

private:
    jit_brgemm_kernel_base_t *brgemm_kernel_ = nullptr;

    DNNL_DISALLOW_COPY_AND_ASSIGN(brgemm_kernel_t);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s