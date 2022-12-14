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

#include "cunumeric/binary/binary_op.h"
#include "cunumeric/binary/binary_op_template.inl"
#include "cunumeric/execution_policy/indexing/parallel_loop_omp.h"

namespace cunumeric {

/*static*/ void BinaryOpTask::omp_variant(TaskContext& context)
{
  binary_op_template<VariantKind::OMP>(context);
}

}  // namespace cunumeric
