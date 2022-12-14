/* Copyright 2021-2022 NVIDIA Corporation
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

// Useful for IDEs
#include "cunumeric/binary/binary_op.h"
#include "cunumeric/binary/binary_op_util.h"
#include "cunumeric/pitches.h"
#include "cunumeric/execution_policy/indexing/parallel_loop.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, BinaryOpCode OP_CODE, LegateTypeCode CODE, int DIM>
struct BinaryOpFunctor {
  using OP    = BinaryOp<OP_CODE, CODE>;
  using TRHS1 = legate_type_of<CODE>;
  using TRHS2 = rhs2_of_binary_op<OP_CODE, CODE>;
  using TLHS  = std::result_of_t<OP(TRHS1, TRHS2)>;
  using OUT   = AccessorWO<TLHS, DIM>;
  using RHS1  = AccessorRO<TRHS1, DIM>;
  using RHS2  = AccessorRO<TRHS2, DIM>;

  OUT out;
  RHS1 in1;
  RHS2 in2;

  TLHS* outptr;
  const TRHS1* in1ptr;
  const TRHS2* in2ptr;

  OP func{};

  Pitches<DIM - 1> pitches;
  Rect<DIM> rect;
  bool dense;
  size_t volume;

  struct DenseTag {};
  struct SparseTag {};

  // constructor

  BinaryOpFunctor(BinaryOpArgs& args) : dense(false), func(args.args)
  {
    rect   = args.out.shape<DIM>();
    volume = pitches.flatten(rect);
    if (volume == 0) return;

    out = args.out.write_accessor<TLHS, DIM>(rect);
    in1 = args.in1.read_accessor<TRHS1, DIM>(rect);
    in2 = args.in2.read_accessor<TRHS2, DIM>(rect);

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in1.accessor.is_dense_row_major(rect) &&
                 in2.accessor.is_dense_row_major(rect);
    if (dense) {
      outptr = out.ptr(rect);
      in1ptr = in1.ptr(rect);
      in2ptr = in2.ptr(rect);
    }
#endif
    // func = OP(args.args);
  }

  __CUDA_HD__ void operator()(const size_t idx, DenseTag) const noexcept
  {
    outptr[idx] = func(in1ptr[idx], in2ptr[idx]);
  }

  __CUDA_HD__ void operator()(const size_t idx, SparseTag) const noexcept
  {
    auto p = pitches.unflatten(idx, rect.lo);
    out[p] = func(in1[p], in2[p]);
  }

  void execute() const noexcept
  {
#ifndef LEGION_BOUNDS_CHECKS
    if (dense) { return ParallelLoopPolicy<KIND, DenseTag>()(rect, *this); }
#endif
    return ParallelLoopPolicy<KIND, SparseTag>()(rect, *this);
  }
};

template <VariantKind KIND, BinaryOpCode OP_CODE>
struct BinaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryOpArgs& args) const
  {
    BinaryOpFunctor<KIND, OP_CODE, CODE, DIM> binaryop(args);
    binaryop.execute();
  }
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!BinaryOp<OP_CODE, CODE>::valid>* = nullptr>
  void operator()(BinaryOpArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
struct BinaryOpDispatch {
  template <BinaryOpCode OP_CODE>
  void operator()(BinaryOpArgs& args) const
  {
    auto dim = std::max(1, args.out.dim());
    double_dispatch(dim, args.in1.code(), BinaryOpImpl<KIND, OP_CODE>{}, args);
  }
};

template <VariantKind KIND>
static void binary_op_template(TaskContext& context)
{
  auto& inputs  = context.inputs();
  auto& outputs = context.outputs();
  auto& scalars = context.scalars();

  std::vector<Store> extra_args;
  for (size_t idx = 2; idx < inputs.size(); ++idx) extra_args.push_back(std::move(inputs[idx]));

  BinaryOpArgs args{
    inputs[0], inputs[1], outputs[0], scalars[0].value<BinaryOpCode>(), std::move(extra_args)};
  op_dispatch(args.op_code, BinaryOpDispatch<KIND>{}, args);
}

}  // namespace cunumeric
