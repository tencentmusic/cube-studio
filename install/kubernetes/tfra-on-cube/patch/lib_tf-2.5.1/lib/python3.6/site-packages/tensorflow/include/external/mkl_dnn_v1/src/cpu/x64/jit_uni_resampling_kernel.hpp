/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef CPU_X64_UNI_RESAMPLING_KERNEL_HPP
#define CPU_X64_UNI_RESAMPLING_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_resampling_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct jit_uni_resampling_kernel : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_resampling)

    jit_uni_resampling_kernel(const jit_resampling_conf_t conf);

    virtual ~jit_uni_resampling_kernel() = default;

protected:
    using Xmm = Xbyak::Xmm;
    using Ymm = Xbyak::Ymm;
    using Zmm = Xbyak::Zmm;
    using Opmask = Xbyak::Opmask;
    using Reg64 = Xbyak::Reg64;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    constexpr int vmm_idx(int idx) const {
        return (cpu_isa_traits<isa>::n_vregs - 1) - idx;
    }

    /*
     * Prepare the mask to be used during tail processing.
     * vmm_tail_mask_ is filled if it is avx and
     * if it is avx512_common at least then k_tail_mask_ is filled.
     */
    void prepare_mask();

    /*
     * Emulates the behavior of vgatherdps for architectures
     * that do not support this instruction.
     */
    void emu_gather_data(const Reg64 &reg_src_addr, const int indices_idx,
            const int data_idx, const bool is_tail = false);

    void gather_data(const Reg64 &reg_src_addr, const int indices_idx,
            const int data_idx, const bool is_tail = false);

    void store_data(const int data_idx, const Reg64 &reg_dst_addr,
            const int offset = 0, const bool is_tail = false);

    void load_data(const Reg64 &reg_src_addr, const int offset,
            const int data_idx, const bool is_tail = false);

    void nearest_ncsp_format();
    void nearest_c_oriented_format();
    void linear_ncsp_format();
    void linear_c_oriented_format();

    void generate() override;

    // Used only for avx and if c tail is present.
    const Vmm vmm_tail_mask_ = Vmm(0);
    // Used only for avx2 and if ncsp format is present.
    // Vgatherdps always gets data using a conditional mask.
    // This register contains all bits set to 1, allowing
    // to get the maximum number of values available to the register
    const Vmm vmm_full_mask_ = Vmm(1);
    const Vmm vmm_src_ = Vmm(2);
    const Vmm vmm_weights_ = Vmm(3);
    const Vmm vmm_indices_ = Vmm(4);
    const Vmm vmm_tmp_ = Vmm(5);

    const Opmask k_tail_mask_ = k1;
    const Opmask k_full_mask_ = k2;

    const Zmm bf16_emu_reserv_1 = Zmm(7);
    const Zmm bf16_emu_reserv_2 = Zmm(8);
    const Zmm bf16_emu_reserv_3 = Zmm(9);
    const Reg64 bf16_emu_scratch = r15;
    const Zmm bf16_emu_reserv_4 = Zmm(10);

    const Reg64 reg_tmp_ = rax;
    const Reg64 reg_dst_ = rbx;
    const Reg64 reg_indices_ = rcx;
    const Reg64 reg_work_ = rdx;
    // Always mimic the Unix ABI
    const Reg64 reg_param = rdi;
    const Reg64 reg_weights = rsi;
    const Reg64 reg_src_ = r8;
    const Reg64 reg_aux_src_0_ = r9;
    const Reg64 reg_aux_src_1_ = r10;
    const Reg64 reg_aux_src_2_ = r11;
    const Reg64 reg_tmp1_ = r15;

    // Registers which are used only for linear algorithm
    // and for channel oriented formats.
    // Meaning of shortcuts:
    // f - front, b - back
    // t - top, b - bottom
    // l - left, r - right
    // Example:
    // src_ftl_ - source tensor data for the front top left corner
    // reg_src_ftl_ - register which contains address of source
    //                tensor data for the front top left corner
    const Vmm weight_left_ = Vmm(1);
    const Vmm weight_right_ = Vmm(2);
    const Vmm weight_top_ = Vmm(3);
    const Vmm weight_bottom_ = Vmm(4);
    const Vmm weight_front_ = Vmm(5);
    const Vmm weight_back_ = Vmm(6);
    const Vmm src_ftl_ = Vmm(vmm_idx(0));
    const Vmm src_ftr_ = Vmm(vmm_idx(1));
    const Vmm src_fbl_ = Vmm(vmm_idx(2));
    const Vmm src_fbr_ = Vmm(vmm_idx(3));
    const Vmm src_btl_ = Vmm(vmm_idx(4));
    const Vmm src_btr_ = Vmm(vmm_idx(5));
    const Vmm src_bbl_ = Vmm(vmm_idx(6));
    const Vmm src_bbr_ = Vmm(vmm_idx(7));

    const Reg64 reg_src_ftl_ = reg_src_;
    const Reg64 reg_src_ftr_ = reg_aux_src_0_;
    const Reg64 reg_src_fbl_ = reg_aux_src_1_;
    const Reg64 reg_src_fbr_ = reg_aux_src_2_;
    const Reg64 reg_src_btl_ = r12;
    const Reg64 reg_src_btr_ = r13;
    const Reg64 reg_src_bbl_ = r14;
    const Reg64 reg_src_bbr_ = r15;

    const jit_resampling_conf_t conf_;
    std::unique_ptr<bf16_emulation_t> bf16_emulation_;
};
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
