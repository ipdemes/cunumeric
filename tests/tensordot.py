# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np

import cunumeric as lg


def test(ty):
    a = lg.random.rand(3, 5, 4).astype(ty)
    b = lg.random.rand(4, 5, 3).astype(ty)

    cn = np.tensordot(a, b, axes=1)
    c = lg.tensordot(a, b, axes=1)

    assert np.allclose(cn, c)

    a = lg.random.rand(3, 5, 4).astype(ty)
    b = lg.random.rand(5, 4, 3).astype(ty)

    cn = np.tensordot(a, b)
    c = lg.tensordot(a, b)

    assert np.allclose(cn, c)

    a = lg.arange(60.0).reshape((3, 4, 5)).astype(ty)
    b = lg.arange(24.0).reshape((4, 3, 2)).astype(ty)

    cn = np.tensordot(a, b, axes=([1, 0], [0, 1]))
    c = lg.tensordot(a, b, axes=([1, 0], [0, 1]))

    assert np.allclose(cn, c)

    a = lg.random.rand(5, 4).astype(ty)
    b = lg.random.rand(4, 5).astype(ty)

    cn = np.tensordot(a, b, axes=1)
    c = lg.tensordot(a, b, axes=1)

    assert np.allclose(cn, c)

    a = lg.random.rand(5, 4).astype(ty)
    b = lg.random.rand(5, 4).astype(ty)

    cn = np.tensordot(a, b)
    c = lg.tensordot(a, b)

    assert np.allclose(cn, c)


if __name__ == "__main__":
    test(np.float16)
    test(np.float32)
    test(np.float64)