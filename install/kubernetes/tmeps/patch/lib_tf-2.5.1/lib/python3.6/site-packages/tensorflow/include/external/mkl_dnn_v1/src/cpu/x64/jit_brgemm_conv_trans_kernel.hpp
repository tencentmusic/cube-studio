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

#ifndef CPU_X64_JIT_BRGEMM_CONV_TRANS_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_CONV_TRANS_KERNEL_HPP

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_brgemm_conv_trans_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_brgemm_conv_trans_kernel_t)

    using reg64_t = const Xbyak::Reg64;

    jit_avx512_core_brgemm_conv_trans_kernel_t(
            const jit_brgemm_conv_conf_t &ajcp);

private:
    jit_brgemm_conv_conf_t jcp;
    dim_t src_dsz;
    dim_t ic_block_sz;
    dim_t iw_size;
    dim_t out_h_offset;
    dim_t VL, n_vec;

    const reg64_t inp_ptr = r15;
    const reg64_t out_ptr = r14;

    const reg64_t aux_inp_ptr = r13;
    const reg64_t aux_out_ptr = r12;

    /* relow stuff */
    const reg64_t reg_kht = r11;
    const reg64_t reg_khp = r10;
    const reg64_t reg_tov = r9;
    const reg64_t reg_bov = r8;
    const reg64_t reg_kwp = rax;
    const reg64_t reg_lov = aux_inp_ptr;
    const reg64_t reg_rov = rbx;
    const reg64_t save_out_ptr = rdx;
    const reg64_t reg_cnt = rbp;
    /* relow stuff */

    /* non-relow stuff */
    const reg64_t khp = r11;
    const reg64_t khc = r10;

    const reg64_t reg_icb = r9;

    const reg64_t kh_over = r8;
    const reg64_t tover = rax;
    const reg64_t bover = rbx;

    const reg64_t reg_owb = rdx;

    const reg64_t reg_tmp = rsi;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask kblock_tail_mask = Xbyak::Opmask(3);

    const Xbyak::Ymm ymm_tmp = Xbyak::Ymm(0);
    const Xbyak::Zmm zmm_tmp = Xbyak::Zmm(0);
    const Xbyak::Zmm zmm_zero = Xbyak::Zmm(1);

    void load(const Xbyak::Xmm &x, const Xbyak::Address &addr);

    void store(const Xbyak::Address &addr, const Xbyak::Xmm &x);

    void zero_ic_block(int icb, dim_t out_off);
    void copy_ic_block(int icb, dim_t inp_off, dim_t out_off);
    void generate() override;
    void copy_row(int icb);
    void copy_row_body(int lpad, int ow_len, int iw_len, int icb);
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
