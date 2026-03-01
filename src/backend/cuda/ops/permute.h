// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/core/tensor.h"
#include <cuda_runtime.h>
#include <vector>

namespace sf {
namespace cuda_ops {

void permute_tensor(const Tensor& src, Tensor& dst, 
                    const std::vector<int64_t>& dims, 
                    cudaStream_t stream);

}  // namespace cuda_ops
}  // namespace sf
