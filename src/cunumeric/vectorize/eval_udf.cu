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

#include "cunumeric/vectorize/eval_udf.h"
#include "cunumeric/vectorize/eval_udf_template.inl"
#include "cunumeric/execution_policy/indexing/parallel_loop.cuh"

namespace cunumeric {

/*static*/ void EvalUdfTask::gpu_variant(TaskContext& context)
{
  eval_udf_template<VariantKind::GPU>(context);
}

template <>
struct UDF<VariantKind::GPU> {
  CUfunction udf;
  UDF() {}
  UDF(int64_t hash, uint64_t) { udf = get_udf(hash); }
  __CUDA_HD__ void call_udf_dense(const size_t idx) const
  {
    printf("IRINA DEBUG inside GPU kernel");
  }
  template <int DIM = 1>
  __CUDA_HD__ void call_udf_sparse(const size_t idx, Point<DIM>& p) const
  {
    printf("IRINA DEBUG inside GPU kernel");
  }
};

}  // namespace cunumeric
