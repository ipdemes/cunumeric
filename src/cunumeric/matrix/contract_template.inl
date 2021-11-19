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

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE>
struct ContractImplBody;

template <LegateTypeCode CODE>
struct support_contract : std::false_type {
};
template <>
struct support_contract<LegateTypeCode::FLOAT_LT> : std::true_type {
};
template <>
struct support_contract<LegateTypeCode::DOUBLE_LT> : std::true_type {
};
template <>
struct support_contract<LegateTypeCode::COMPLEX64_LT> : std::true_type {
};
template <>
struct support_contract<LegateTypeCode::COMPLEX128_LT> : std::true_type {
};

template <VariantKind KIND>
struct ContractImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<support_contract<CODE>::value>* = nullptr>
  void operator()(ContractArgs& args) const
  {
    using T = legate_type_of<CODE>;

    std::vector<int64_t> lhs_shape;
    std::vector<int64_t> lhs_strides;
    std::vector<int32_t> lhs_modes;
    Rect<DIM> lhs_bloated_shape = args.lhs.shape<DIM>();
    size_t lhs_bloated_strides[DIM];
    T* lhs_data = args.lhs.reduce_accessor<SumReduction<T>, true, DIM>(lhs_bloated_shape)
                    .ptr(lhs_bloated_shape, lhs_bloated_strides);

    for (int i = 0; i < DIM; ++i) {
      if (!args.lhs_dim_mask[i]) { continue; }
      lhs_shape.push_back(lhs_bloated_shape.hi[i] - lhs_bloated_shape.lo[i] + 1);
      lhs_strides.push_back(lhs_bloated_strides[i]);
      lhs_modes.push_back(i + 'a');
    }

    std::vector<int64_t> rhs1_shape;
    std::vector<int64_t> rhs1_strides;
    std::vector<int32_t> rhs1_modes;
    Rect<DIM> rhs1_bloated_shape = args.rhs1.shape<DIM>();
    size_t rhs1_bloated_strides[DIM];
    const T* rhs1_data = args.rhs1.read_accessor<T, DIM>(rhs1_bloated_shape)
                           .ptr(rhs1_bloated_shape, rhs1_bloated_strides);
    for (int i = 0; i < DIM; ++i) {
      if (!args.rhs1_dim_mask[i]) { continue; }
      rhs1_shape.push_back(rhs1_bloated_shape.hi[i] - rhs1_bloated_shape.lo[i] + 1);
      rhs1_strides.push_back(rhs1_bloated_strides[i]);
      rhs1_modes.push_back(i + 'a');
    }

    std::vector<int64_t> rhs2_shape;
    std::vector<int64_t> rhs2_strides;
    std::vector<int32_t> rhs2_modes;
    Rect<DIM> rhs2_bloated_shape = args.rhs2.shape<DIM>();
    size_t rhs2_bloated_strides[DIM];
    const T* rhs2_data = args.rhs2.read_accessor<T, DIM>(rhs2_bloated_shape)
                           .ptr(rhs2_bloated_shape, rhs2_bloated_strides);
    for (int i = 0; i < DIM; ++i) {
      if (!args.rhs2_dim_mask[i]) { continue; }
      rhs2_shape.push_back(rhs2_bloated_shape.hi[i] - rhs2_bloated_shape.lo[i] + 1);
      rhs2_strides.push_back(rhs2_bloated_strides[i]);
      rhs2_modes.push_back(i + 'a');
    }

    ContractImplBody<KIND, CODE>()(lhs_data,
                                   lhs_shape.size(),
                                   lhs_shape.data(),
                                   lhs_strides.data(),
                                   lhs_modes.data(),
                                   rhs1_data,
                                   rhs1_shape.size(),
                                   rhs1_shape.data(),
                                   rhs1_strides.data(),
                                   rhs1_modes.data(),
                                   rhs2_data,
                                   rhs2_shape.size(),
                                   rhs2_shape.data(),
                                   rhs2_strides.data(),
                                   rhs2_modes.data());
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!support_contract<CODE>::value>* = nullptr>
  void operator()(ContractArgs& args) const
  {
    assert(false);
  }
};

template <VariantKind KIND>
static void contract_template(legate::TaskContext& context)
{
  auto& reductions = context.reductions();
  auto& inputs     = context.inputs();
  auto& scalars    = context.scalars();

  ContractArgs args{reductions[0],
                    inputs[0],
                    inputs[1],
                    scalars[0].values<const bool>(),
                    scalars[1].values<const bool>(),
                    scalars[2].values<const bool>()};

  auto dim  = args.lhs.dim();
  auto code = args.lhs.code();

#ifdef DEBUG_CUNUMERIC
  assert(dim = args.rhs1.dim());
  assert(dim = args.rhs2.dim());
  assert(dim = args.lhs_dim_mask.size());
  assert(dim = args.rhs1_dim_mask.size());
  assert(dim = args.rhs2_dim_mask.size());
  assert(code == args.rhs1.code());
  assert(code == args.rhs2.code());
#endif

  double_dispatch(dim, code, ContractImpl<KIND>{}, args);
}

}  // namespace cunumeric
