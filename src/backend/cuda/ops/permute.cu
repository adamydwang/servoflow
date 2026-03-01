// SPDX-License-Identifier: Apache-2.0
#include "permute.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <iostream>

namespace sf {
namespace cuda_ops {

// 5-dim coordinate conversion
// coords[i] = idx % shape[i]; idx /= shape[i];
// But shape is [N-1...0] order for contiguous.
// dst is contiguous.
// Let's assume dst.shape is passed as d0, d1, d2, d3, d4.
// idx = i0*S0 + i1*S1 + ...
// We need to decode idx to (i0, i1, i2, i3, i4).
// Since dst is contiguous row-major:
// i4 = idx % d4; idx /= d4;
// i3 = idx % d3; idx /= d3;
// ...
// Correct.

template<typename T>
__global__ void permute_kernel_impl(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int64_t num_elements,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,
    int ndim)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int64_t coords[5] = {0, 0, 0, 0, 0};
    int64_t temp = idx;
    
    // Decode output index (assuming dst is contiguous row-major)
    if (ndim >= 5) { coords[4] = temp % d4; temp /= d4; }
    if (ndim >= 4) { coords[3] = temp % d3; temp /= d3; }
    if (ndim >= 3) { coords[2] = temp % d2; temp /= d2; }
    if (ndim >= 2) { coords[1] = temp % d1; temp /= d1; }
    if (ndim >= 1) { coords[0] = temp; }

    // Compute input offset: sum(coords[i] * strides[i])
    // strides are permuted src strides.
    // e.g. dst dim 0 corresponds to src dim k. strides[0] = src.stride(k).
    int64_t src_offset = 0;
    if (ndim >= 1) src_offset += coords[0] * s0;
    if (ndim >= 2) src_offset += coords[1] * s1;
    if (ndim >= 3) src_offset += coords[2] * s2;
    if (ndim >= 4) src_offset += coords[3] * s3;
    if (ndim >= 5) src_offset += coords[4] * s4;

    dst[idx] = src[src_offset];
}

void permute_tensor(const Tensor& src, Tensor& dst, 
                    const std::vector<int64_t>& dims, 
                    cudaStream_t stream) 
{
    int64_t numel = src.numel();
    if (numel == 0) return;
    
    int ndim = static_cast<int>(src.ndim());
    if (ndim > 5) throw std::runtime_error("permute: supports up to 5 dims");

    dim3 block(256);
    dim3 grid((static_cast<unsigned int>(numel) + block.x - 1) / block.x);

    // Prepare permuted strides
    // p_strides[i] = src_stride[dims[i]]
    
    const auto& src_strides = src.strides();
    const auto& dst_shape = dst.shape(); 
    
    int64_t s[5] = {0};
    int64_t d[5] = {1, 1, 1, 1, 1};

    for (int i = 0; i < ndim; ++i) {
        s[i] = src_strides[dims[i]];
        d[i] = dst_shape[i];
    }
    
    // Dispatch
    if (src.dtype() == DType::Float16) {
        permute_kernel_impl<half><<<grid, block, 0, stream>>>(
            static_cast<const half*>(src.raw_data_ptr()), 
            static_cast<half*>(dst.raw_data_ptr()), 
            numel,
            d[0], d[1], d[2], d[3], d[4],
            s[0], s[1], s[2], s[3], s[4],
            ndim);
    } else if (src.dtype() == DType::Float32) {
        permute_kernel_impl<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(src.raw_data_ptr()), 
            static_cast<float*>(dst.raw_data_ptr()), 
            numel,
            d[0], d[1], d[2], d[3], d[4],
            s[0], s[1], s[2], s[3], s[4],
            ndim);
    } else if (src.dtype() == DType::Int8) {
        permute_kernel_impl<int8_t><<<grid, block, 0, stream>>>(
            static_cast<const int8_t*>(src.raw_data_ptr()), 
            static_cast<int8_t*>(dst.raw_data_ptr()), 
            numel,
            d[0], d[1], d[2], d[3], d[4],
            s[0], s[1], s[2], s[3], s[4],
            ndim);
    } else {
        throw std::runtime_error("permute: unsupported dtype");
    }
}

} // namespace cuda_ops
} // namespace sf
