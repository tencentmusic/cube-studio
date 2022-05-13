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

#ifndef CPU_X64_MATMUL_BRGEMM_MATMUL_UTILS_HPP
#define CPU_X64_MATMUL_BRGEMM_MATMUL_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

struct brgemm_matmul_conf_t {
    int ndims, batch_ndims;
    dim_t M, N, K, batch;
    dim_t M_blk, N_blk, K_blk, M_tail, N_tail, K_tail;
    int M_chunk_size, N_chunk_size;
    dim_t LDA, LDB, LDC, LDD;
    int brgemm_batch_size;
    int wei_n_blk, wei_k_blk;
    brgemm_batch_kind_t brg_type;

    cpu_isa_t isa;

    format_tag_t src_tag, wei_tag, dst_tag, bia_tag;
    bool use_buffer;
    bool with_bias;
    bool with_sum;
    bool with_eltwise;
    bool with_scales;
    bool signed_input;
    bool is_oscale_per_n;
    bool use_buffer_b;
    bool use_buffer_a;
    bool use_buffer_a_tail_only;
    post_ops_t::entry_t::eltwise_t eltwise;

    data_type_t src_dt;
    data_type_t dst_dt;
    data_type_t wei_dt;
    data_type_t acc_dt;
    data_type_t bia_dt;
    int nthr;
};

status_t init_brgemm_matmul_conf(cpu_isa_t isa, brgemm_matmul_conf_t &bgmmc,
        const matmul_desc_t &mmd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr);

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const brgemm_matmul_conf_t &bgmmc);

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
