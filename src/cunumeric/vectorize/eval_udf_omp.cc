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
#include "cunumeric/execution_policy/indexing/parallel_loop_omp.h"

namespace cunumeric {

/*static*/ void EvalUdfTask::omp_variant(TaskContext& context)
{
  eval_udf_template<VariantKind::OMP>(context);
}

template <>
struct UDF<VariantKind::OMP> {
  using UDFT = void(void**);
  UDFT* udf;
  UDF() {}
  UDF(int64_t, uint64_t cpu_func_ptr) { udf = reinterpret_cast<UDFT*>(cpu_func_ptr); }
  __CUDA_HD__ void call_udf_dense(const size_t idx) const
  {
    printf("IRINA DEBUG inside OMP kernel");
    std::vector<void*> udf_args;
    // for (auto& out : outptrs) { udf_args.push_back(reinterpret_cast<void*>(&outptr[idx])); }
    // for (auto& in : inptrs) {
    //   udf_args.push_back(reinterpret_cast<void*>(const_cast<T*>(&inptr[idx])));
    // }
    // for (auto s : scalars) { udf_args.push_back(const_cast<void*>(s.ptr())); }
    udf(udf_args.data());
  }
  template <int DIM = 1>
  __CUDA_HD__ void call_udf_sparse(const size_t idx, Point<DIM>& p) const
  {
    printf("IRINA DEBUG inside OMP kernel");
    std::vector<void*> udf_args;
    // for (auto& out : outputs) { udf_args.push_back(reinterpret_cast<void*>(&out[p])); }
    // for (auto& in : inputs) {
    // udf_args.push_back(reinterpret_cast<void*>(const_cast<T*>(&in[p]))); } for (auto s : scalars)
    // { udf_args.push_back(const_cast<void*>(s.ptr())); }
    udf(udf_args.data());
  }
};
}  // namespace cunumeric
