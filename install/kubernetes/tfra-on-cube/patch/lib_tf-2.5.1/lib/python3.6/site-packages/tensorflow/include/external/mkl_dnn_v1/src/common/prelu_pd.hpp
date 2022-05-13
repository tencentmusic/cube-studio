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

#ifndef COMMON_PRELU_PD_HPP
#define COMMON_PRELU_PD_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/primitive_desc.hpp"
#include "dnnl.h"

namespace dnnl {
namespace impl {

struct prelu_fwd_pd_t;

struct prelu_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::prelu;

    prelu_pd_t(const prelu_desc_t *adesc, const primitive_attr_t *attr,
            const prelu_fwd_pd_t *hint_fwd_pd);

    const prelu_desc_t *desc() const;
    const op_desc_t *op_desc() const override;

    status_t query(query_t what, int idx, void *result) const override;

    /* common prelu aux functions */
    dim_t N() const;
    dim_t C() const;
    dim_t D() const;
    dim_t H() const;
    dim_t W() const;

    int ndims() const;
    bool has_zero_dim_memory() const;
    bool is_fwd() const;
    const memory_desc_t *weights_md(int index) const override;
    const memory_desc_t *src_md(int index) const override;
    const memory_desc_t *dst_md(int index) const override;
    size_t dtype_size() const;

protected:
    prelu_desc_t desc_;
    const prelu_fwd_pd_t *hint_fwd_pd_;
    memory_desc_t data_md_;
    memory_desc_t weights_md_;

private:
    const memory_desc_t &data_desc() const;
};

struct prelu_fwd_pd_t : public prelu_pd_t {
    typedef prelu_fwd_pd_t hint_class;

    prelu_fwd_pd_t(const prelu_desc_t *adesc, const primitive_attr_t *attr,
            const prelu_fwd_pd_t *hint_fwd_pd);

    arg_usage_t arg_usage(int arg) const override;

    const memory_desc_t *arg_md(int arg) const override;

    int n_inputs() const override;
    int n_outputs() const override;

protected:
    bool set_default_formats();
};

struct prelu_bwd_pd_t : public prelu_pd_t {
    typedef prelu_fwd_pd_t hint_class;

    prelu_bwd_pd_t(const prelu_desc_t *adesc, const primitive_attr_t *attr,
            const prelu_fwd_pd_t *hint_fwd_pd);

    arg_usage_t arg_usage(int arg) const override;
    const memory_desc_t *arg_md(int arg) const override;
    const memory_desc_t *diff_src_md(int index) const override;
    const memory_desc_t *diff_dst_md(int index) const override;
    const memory_desc_t *diff_weights_md(int index) const override;
    int n_inputs() const override;
    int n_outputs() const override;

protected:
    memory_desc_t diff_data_md_;
    memory_desc_t diff_weights_md_;

    bool set_default_formats();
};

} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
