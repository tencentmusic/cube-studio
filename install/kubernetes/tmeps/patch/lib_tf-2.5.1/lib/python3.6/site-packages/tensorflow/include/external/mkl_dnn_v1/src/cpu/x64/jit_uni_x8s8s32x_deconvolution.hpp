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

#ifndef CPU_X64_JIT_UNI_X8S8S32X_DECONVOLUTION_HPP
#define CPU_X64_JIT_UNI_X8S8S32X_DECONVOLUTION_HPP

#include <memory>
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_deconvolution_pd.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

template <cpu_isa_t isa, typename Vmm>
struct _jit_uni_x8s8s32x_deconv_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_uni_x8s8s32x_deconv_fwd_kernel);

    _jit_uni_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_wrapper &dst_d);
    ~_jit_uni_x8s8s32x_deconv_fwd_kernel();
    const jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa, Vmm>>
            postops_injector_;
    using reg64_t = const Xbyak::Reg64;

    enum {
        ic_sub_step = 4,
        ker_max_reg_idx = 13,
    };
    enum ker_block_t {
        no_last_block = 0x1U,
        last_ic_block = 0x2U,
        last_sp_block = 0x4U,
    };

    /* data regs */
    reg64_t reg_src = r8;
    reg64_t reg_filt = r9;
    reg64_t reg_dst = r10;
    reg64_t param1 = abi_param1;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_ki = r14;

    reg64_t reg_nur_w = rbx;
    reg64_t reg_bias = rdx;
    reg64_t reg_icb = reg_bias;
    reg64_t reg_ptr_scales = rax;
    reg64_t reg_ptr_saturation_ubound = rax;
    reg64_t reg_oc_blocks = rsi;

    reg64_t aux_reg_src = r11;
    reg64_t aux_reg_filt = r12;

    reg64_t aux_reg_src_d = r13;
    reg64_t aux_reg_filt_d = r15;

    reg64_t reg_compensation = r14;
    reg64_t reg_scratch = r14;
    reg64_t reg_ptr_sum_scale = r11;
    reg64_t reg_bias_alpha = abi_not_param1;
    reg64_t reg_overflow = rax;
    reg64_t reg_comp_strides = reg_overflow;
    reg64_t reg_ker_long_offt = r15;

    const Vmm vmm_tmp = Vmm(3);
    const Vmm vmm_one = Vmm(2);
    /* used during write-out section of store_output */
    const Vmm vmm_zero = Vmm(0);
    const Vmm vmm_saturation = Vmm(0);
    const Vmm vmm_wei = Vmm(0);
    const Vmm vmm_scale = Vmm(0);
    const Vmm vmm_mask = Vmm(2);
    /* signed input */
    const Vmm vmm_shift = Vmm(1);
    const Vmm vmm_comp = Vmm(1);
    const Vmm vmm_bias = Vmm(0);
    const Vmm vmm_prev_dst = Vmm(0);

    const Vmm vmm_out(int i_ur, int i_oc) {
        int idx = i_ur * jcp.nb_oc_blocking + i_oc;
        assert(idx < ker_max_reg_idx);
        /* remap the reg indices to avoid using xmm0 in eltwise injector */
        return Vmm(15 - idx);
    }
    const Vmm vmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < ker_max_reg_idx);
        return Vmm(15 - idx);
    }
    const Vmm vmm_bias_alpha() {
        return Vmm(15 - jcp.nb_oc_blocking * jcp.ur_w);
    }
    Xmm xmm_bias_alpha() { return Xmm(15 - jcp.nb_oc_blocking * jcp.ur_w); }

    int get_ow_start(int ki, int l_overflow) {
        int res = (jcp.ow - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;
        return res;
    }

    int get_ow_end(int ur_w, int ki, int r_overflow) {
        if (utils::one_of(ur_w, jcp.ow, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;
        return ur_w - res;
    }

    int get_blocking_size() {
        return jcp.is_depthwise ? jcp.ch_block : jcp.oc_block;
    }
    int get_tail_size() {
        return jcp.is_depthwise ? jcp.ngroups % jcp.ch_block
                                : jcp.oc_without_padding % jcp.oc_block;
    }

    void prepare_output(int ur_w);
    void apply_postops(int ur_w, bool last_oc_block, const float *p_sum_scale);
    void store_output(int ur_w, bool last_oc_block);
    void compute_ker(int ur_w, int l_overflow, int r_overflow,
            ker_block_t last_ic_block_flag, bool h_padded = false);
    void kh_loop(int ur_w, int pad_l, int pad_r, ker_block_t last_ker_block);
    void icb_loop(int ur_w, int pad_l, int pad_r, bool last_block);
    void generate() override;
    void cvt2ps(data_type_t type_in, const Vmm &vmm_in, const Reg64 &reg,
            int offset, int load_size);
};

template <cpu_isa_t isa>
struct jit_uni_x8s8s32x_deconv_fwd_kernel {

    jit_uni_x8s8s32x_deconv_fwd_kernel(const jit_conv_conf_t &ajcp,
            const primitive_attr_t &attr, const memory_desc_wrapper &dst_d)
        : kernel_(nullptr) {

        int ch_block = ajcp.is_depthwise ? ajcp.ch_block : ajcp.ic_block;
        switch (ch_block) {
            case 8:
                if (isa == avx2) {
                    kernel_ = new _jit_avx2_x8s8s32x_deconv_fwd_kernel(
                            ajcp, attr, dst_d);
                    return;
                } else
                    assert(!"invalid channel blocking for current ISA");
            case 4:
                kernel_ = new _jit_uni_x8s8s32x_deconv_fwd_kernel<isa,
                        Xbyak::Xmm>(ajcp, attr, dst_d);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() { return kernel_->create_kernel(); }

    ~jit_uni_x8s8s32x_deconv_fwd_kernel() { delete kernel_; }

    void operator()(const jit_deconv_call_s *p) const { (*kernel_)(p); }

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const deconvolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            const bool with_bias, memory_desc_t &bias_md,
            const primitive_attr_t &attr, int nthreads);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    using _jit_avx2_x8s8s32x_deconv_fwd_kernel
            = _jit_uni_x8s8s32x_deconv_fwd_kernel<avx2, Xbyak::Ymm>;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_uni_x8s8s32x_deconv_fwd_kernel);
    jit_generator *kernel_;
};

template <cpu_isa_t isa, impl::data_type_t src_type, impl::data_type_t dst_type>
struct _jit_uni_x8s8s32x_deconvolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        using cpu_deconvolution_fwd_pd_t::cpu_deconvolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_uni_deconv:",
                        isa == avx2 && jcp_.ver == ver_vnni ? avx2_vnni : isa,
                        ""),
                _jit_uni_x8s8s32x_deconvolution_fwd_t);

        status_t init(engine_t *engine) {
            bool ok = true && is_fwd()
                    && (desc()->alg_kind & alg_kind::deconvolution_direct)
                    && desc()->src_desc.data_type == src_type
                    && desc()->dst_desc.data_type == dst_type
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type,
                                    data_type::f32, data_type::s32,
                                    data_type::s8, data_type::u8))
                    && desc()->accum_data_type == data_type::s32
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale
                            | primitive_attr_t::skip_mask_t::post_ops);
            if (!ok) return status::unimplemented;

            status_t status
                    = jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::init_conf(jcp_,
                            *desc(), src_md_, weights_md_, dst_md_, with_bias(),
                            bias_md_, *attr(), dnnl_get_max_threads());

            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_uni_x8s8s32x_deconv_fwd_kernel<isa>::init_scratchpad(
                    scratchpad, jcp_, *attr());

            return status::success;
        }

        jit_conv_conf_t jcp_;
    };

    _jit_uni_x8s8s32x_deconvolution_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_uni_x8s8s32x_deconv_fwd_kernel<isa>(pd()->jcp_,
                        *pd()->attr(), memory_desc_wrapper(pd()->dst_md()))));
        return kernel_->create_kernel();
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        const auto &_pd = pd();
        auto ndims = _pd->ndims();
        if (ndims == 3)
            execute_forward_1d(ctx);
        else if (ndims == 4)
            execute_forward_2d(ctx);
        else if (ndims == 5)
            execute_forward_3d(ctx);
        else
            return status::unimplemented;
        return status::success;
    }

private:
    void execute_forward_1d(const exec_ctx_t &ctx) const;
    void execute_forward_2d(const exec_ctx_t &ctx) const;
    void execute_forward_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<jit_uni_x8s8s32x_deconv_fwd_kernel<isa>> kernel_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
