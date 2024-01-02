/* Copyright 2024 NVIDIA Corporation
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
 *
 */

#pragma once

#pragma once

// Useful for IDEs
#include <core/utilities/typedefs.h>
#include "cunumeric/vectorize/eval_udf.h"
#include "cunumeric/pitches.h"
#include "cunumeric/execution_policy/indexing/parallel_loop.h"

namespace cunumeric {

using namespace legate;

template <VariantKind KIND>
struct UDF {};

#if 0
template <VariantKind KIND>
call_udf_dense(UDF<KIND> &udf, const size_t idx)
{
  std::vector<void*> udf_args;
    //for (auto& out : outptrs) { udf_args.push_back(reinterpret_cast<void*>(&outptr[idx])); }
    //for (auto& in : inptrs) {
    //  udf_args.push_back(reinterpret_cast<void*>(const_cast<T*>(&inptr[idx])));
    //}
    //for (auto s : scalars) { udf_args.push_back(const_cast<void*>(s.ptr())); }
  udf(udf_args.data());
}

template <VariantKind KIND, int DIM=1>
call_udf_sparse(UDF<KIND> &udf, const size_t idx, Point<DIM> &p)
{
    std::vector<void*> udf_args;
    //for (auto& out : outputs) { udf_args.push_back(reinterpret_cast<void*>(&out[p])); }
    //for (auto& in : inputs) { udf_args.push_back(reinterpret_cast<void*>(const_cast<T*>(&in[p]))); }
    //for (auto s : scalars) { udf_args.push_back(const_cast<void*>(s.ptr())); }
    udf(udf_args.data());
}
#endif

template <VariantKind KIND, Type::Code CODE, int DIM>
struct EvalUdf {
  using T   = legate_type_of<CODE>;
  using IN  = AccessorRO<T, DIM>;
  using OUT = AccessorRW<T, DIM>;
  // using UDF = void(void**);

  std::vector<IN> inputs;
  std::vector<OUT> outputs;
  std::vector<const T*> inptrs;
  std::vector<T*> outptrs;
  Pitches<DIM - 1> pitches;
  Rect<DIM> rect;
  size_t strides[DIM];
  bool dense;
  size_t volume;
  std::vector<legate::Scalar> scalars;
  UDF<KIND> udf;
  // Legion::Processor point;
  int64_t hash = 0;

  struct DenseTag {};
  struct SparseTag {};

  EvalUdf(EvalUdfArgs& args) : dense(false)
  {
    udf = UDF<KIND>(args.hash, args.cpu_func_ptr);
    // hash   = args.hash;
    //  if (hash!=0){
    //     udf =
    //     cu_udf   = get_udf(args.hash);
    // }
    // udf    = reinterpret_cast<UDF*>(args.cpu_func_ptr);
    volume = 1;
    if (args.inputs.size() > 0) {
      rect   = args.inputs[0].shape<DIM>();
      volume = pitches.flatten(rect);
      // if (rect.empty()) return;
      for (size_t i = 0; i < args.inputs.size(); i++) {
#ifndef LEGATE_BOUNDS_CHECKS
        if (i == 0) {
          auto in = args.inputs[i].read_accessor<T, DIM>(rect);
          dense   = in.accessor.is_dense_row_major(rect);
        }
#endif
        if (i < args.num_outputs) {
          auto out = args.outputs[i].read_write_accessor<T, DIM>(rect);
#ifndef LEGATE_BOUNDS_CHECKS
          dense = dense && out.accessor.is_dense_row_major(rect);
          if (dense) { outptrs.push_back(out.ptr(rect)); }
#endif
          outputs.push_back(out);
        } else {
          auto in = args.inputs[i].read_accessor<T, DIM>(rect);
#ifndef LEGATE_BOUNDS_CHECKS
          dense = dense && in.accessor.is_dense_row_major(rect);
          if (dense) { inptrs.push_back(in.ptr(rect)); }

#endif
          inputs.push_back(in);
        }
      }
    }
    for (auto s : args.scalars) scalars.push_back(s);
    std::cout << "IRINA DEBUG dense = " << dense << ", volume = " << volume << std::endl;
  }  // constructor

  __CUDA_HD__ void operator()(const size_t idx, DenseTag) const noexcept
  {
    udf.call_udf_dense(idx);
#if 0
    std::vector<void*> udf_args;
    //for (auto& out : outptrs) { udf_args.push_back(reinterpret_cast<void*>(&out[idx])); }
    //for (auto& in : inptrs) {
    //  udf_args.push_back(reinterpret_cast<void*>(const_cast<T*>(&in[idx])));
    //}
    //for (auto s : scalars) { udf_args.push_back(const_cast<void*>(s.ptr())); }
    udf(udf_args.data());
#endif
  }

  __CUDA_HD__ void operator()(const size_t idx, SparseTag) const noexcept
  {
    auto p = pitches.unflatten(idx, rect.lo);
    udf.call_udf_sparse(idx, p);
#if 0
    std::vector<void*> udf_args;
    //for (auto& out : outputs) { udf_args.push_back(reinterpret_cast<void*>(&out[p])); }
    //for (auto& in : inputs) { udf_args.push_back(reinterpret_cast<void*>(const_cast<T*>(&in[p]))); }
    //for (auto s : scalars) { udf_args.push_back(const_cast<void*>(s.ptr())); }
    udf(udf_args.data());
#endif
  }

  void execute() const noexcept
  {
#ifndef LEGATE_BOUNDS_CHECKS
    if (dense) { return ParallelLoopPolicy<KIND, DenseTag>()(volume, *this); }
#endif
    return ParallelLoopPolicy<KIND, SparseTag>()(volume, *this);
  }
};

template <VariantKind KIND>
struct EvalUdfImpl {
  template <Type::Code CODE, int DIM>
  void operator()(EvalUdfArgs& args) const
  {
    EvalUdf<KIND, CODE, DIM> eval_udf(args);
    eval_udf.execute();
  }
};

template <VariantKind KIND>
static void eval_udf_template(TaskContext& context)
{
  uint32_t num_outputs = context.scalars()[0].value<uint32_t>();
  uint32_t num_scalars = context.scalars()[1].value<uint32_t>();
  std::vector<Scalar> scalars;
  for (size_t i = 2; i < (2 + num_scalars); i++) scalars.push_back(context.scalars()[i]);

  int64_t ptx_hash = 0;
  if (context.scalars().size() > (3 + num_scalars)) {
    ptx_hash = context.scalars()[3 + num_scalars].value<int64_t>();
  }

  EvalUdfArgs args{context.scalars()[2 + num_scalars].value<uint64_t>(),
                   context.inputs(),
                   context.outputs(),
                   scalars,
                   num_outputs,
                   legate::Processor::get_executing_processor(),
                   ptx_hash};
  int dim = 1;
  if (args.inputs.size() > 0) {
    dim = args.inputs[0].dim() == 0 ? 1 : args.inputs[0].dim();
    assert(dim > 0);
    double_dispatch(dim, args.inputs[0].code(), EvalUdfImpl<KIND>{}, args);
  } else {
    Type::Code code = Type::Code::BOOL;
    double_dispatch(dim, code, EvalUdfImpl<KIND>{}, args);
  }
}

}  // namespace cunumeric
