/* Copyright 2021 NVIDIA Corporation
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

#include "item/read.h"
#include "item/read_template.inl"

namespace legate {
namespace numpy {

using namespace Legion;

template <typename VAL>
static __global__ void __launch_bounds__(1, 1)
  read_value(DeferredValue<VAL> value, const AccessorRO<VAL, 1> in)
{
  value = in[0];
}

template <typename VAL>
struct ReadImplBody<VariantKind::GPU, VAL> {
  UntypedScalar operator()(AccessorRO<VAL, 1> in) const
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    DeferredValue<VAL> result(VAL{0});
    read_value<VAL><<<1, 1, 0, stream>>>(result, in);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return UntypedScalar(result.read());
  }
};

/*static*/ UntypedScalar ReadTask::gpu_variant(const Task *task,
                                               const std::vector<PhysicalRegion> &regions,
                                               Context context,
                                               Runtime *runtime)
{
  return read_template<VariantKind::GPU>(task, regions, context, runtime);
}

}  // namespace numpy
}  // namespace legate
