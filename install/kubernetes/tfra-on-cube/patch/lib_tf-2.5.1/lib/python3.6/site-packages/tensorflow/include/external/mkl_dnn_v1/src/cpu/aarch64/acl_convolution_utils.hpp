/*******************************************************************************
* Copyright 2020-2021 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP
#define CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/cpu_engine.hpp"

#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <typename NEConv>
struct acl_obj_t {
    NEConv conv;
    arm_compute::NEArithmeticAddition add;
    arm_compute::NEActivationLayer act;
    arm_compute::Tensor src_tensor;
    arm_compute::Tensor wei_tensor;
    arm_compute::Tensor bia_tensor;
    arm_compute::Tensor dst_tensor;
    arm_compute::Tensor dst_acc_tensor;
};

struct acl_conv_conf_t {
    bool with_bias;
    bool is_int8;
    bool sum_with_eltwise;
    arm_compute::TensorInfo src_info;
    arm_compute::TensorInfo wei_info;
    arm_compute::TensorInfo bia_info;
    arm_compute::TensorInfo dst_info;
    arm_compute::PadStrideInfo padstride_info;
    arm_compute::Size2D dilation_info;
    arm_compute::WeightsInfo weights_info;
    arm_compute::ActivationLayerInfo act_info;
};

namespace acl_convolution_utils {

status_t init_conf_gemm(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

status_t init_conf_indirect_gemm(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

status_t init_conf_wino(acl_conv_conf_t &acp, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const convolution_desc_t &cd,
        const primitive_attr_t &attr);

arm_compute::DataType get_acl_data_t(const dnnl_data_type_t dt);
arm_compute::ActivationLayerInfo get_acl_act(const primitive_attr_t &attr);
bool acl_act_ok(alg_kind_t eltwise_activation);

} // namespace acl_convolution_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_ACL_CONVOLUTION_UTILS_HPP
