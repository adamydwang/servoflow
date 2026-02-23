// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "servoflow/backend/backend.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sf {
namespace cuda {

// ─────────────────────────────────────────────────────────────────────────────
// CUDAMemoryPool: a simple free-list pool to avoid cudaMalloc/cudaFree on
// every allocation (which synchronises the device).
//
// Strategy: blocks are bucketed by size (rounded up to next power of two).
// Freed blocks are returned to the pool; the pool can be emptied explicitly.
// ─────────────────────────────────────────────────────────────────────────────
class CUDAMemoryPool {
public:
    explicit CUDAMemoryPool(int device_index);
    ~CUDAMemoryPool();

    // Allocate at least `bytes` bytes on the device. Returns device pointer.
    void* alloc(size_t bytes);

    // Return ptr (allocated with this pool) back to the free-list.
    void  free(void* ptr, size_t bytes);

    // Release all cached blocks back to CUDA.
    void  empty_cache();

    size_t cached_bytes()    const;
    size_t allocated_bytes() const;

private:
    struct Block {
        void*  ptr;
        size_t size;
    };

    int device_index_;
    std::unordered_map<size_t, std::vector<Block>> free_blocks_;  // size → list
    size_t cached_bytes_    = 0;
    size_t allocated_bytes_ = 0;
    mutable std::mutex mu_;

    static size_t round_up(size_t bytes);
};

// ─────────────────────────────────────────────────────────────────────────────
// CUDAGraph: wraps a captured cudaGraph_t for replaying a fixed workload.
// ─────────────────────────────────────────────────────────────────────────────
class CUDAGraph {
public:
    CUDAGraph() = default;
    ~CUDAGraph();

    void begin_capture(cudaStream_t stream);
    void end_capture(cudaStream_t stream);
    void launch(cudaStream_t stream);
    bool is_captured() const { return exec_ != nullptr; }

private:
    cudaGraph_t     graph_ = nullptr;
    cudaGraphExec_t exec_  = nullptr;
};

// ─────────────────────────────────────────────────────────────────────────────
// CUDABackend: concrete IBackend implementation for NVIDIA GPUs.
// ─────────────────────────────────────────────────────────────────────────────
class CUDABackend : public IBackend {
public:
    explicit CUDABackend(int device_index);
    ~CUDABackend() override;

    // ── IBackend interface ─────────────────────────────────────────────────
    DeviceType  device_type() const override { return DeviceType::CUDA; }
    BackendCaps caps()        const override;

    Tensor alloc(Shape shape, DType dtype,
                 StreamHandle stream = nullptr) override;
    Tensor alloc_pinned(Shape shape, DType dtype) override;
    void   empty_cache() override;

    void copy(Tensor& dst, const Tensor& src,
              StreamHandle stream = nullptr) override;
    void fill(Tensor& dst, float value,
              StreamHandle stream = nullptr) override;

    StreamHandle create_stream()               override;
    void         destroy_stream(StreamHandle)  override;
    void         sync_stream(StreamHandle)     override;
    void         sync_device()                 override;

    void gemm(const Tensor& A, const Tensor& B, Tensor& C,
              float alpha, float beta,
              bool trans_a, bool trans_b,
              StreamHandle stream) override;

    void batched_gemm(const Tensor& A, const Tensor& B, Tensor& C,
                      float alpha, float beta,
                      bool trans_a, bool trans_b,
                      StreamHandle stream) override;

    void attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                   Tensor& out,
                   const Tensor* mask,
                   float scale, bool is_causal,
                   StreamHandle stream) override;

    void layer_norm(const Tensor& x,
                    const Tensor& gamma, const Tensor& beta,
                    Tensor& out, float eps,
                    StreamHandle stream) override;

    void rms_norm(const Tensor& x, const Tensor& gamma,
                  Tensor& out, float eps,
                  StreamHandle stream) override;

    void add(const Tensor& a, const Tensor& b, Tensor& out,
             StreamHandle stream) override;
    void mul(const Tensor& a, const Tensor& b, Tensor& out,
             StreamHandle stream) override;
    void scale(const Tensor& a, float scalar, Tensor& out,
               StreamHandle stream) override;

    void gelu(const Tensor& x, Tensor& out, StreamHandle stream) override;
    void silu(const Tensor& x, Tensor& out, StreamHandle stream) override;
    void softmax(const Tensor& x, Tensor& out, int64_t dim,
                 StreamHandle stream) override;

    void embedding(const Tensor& weight, const Tensor& indices,
                   Tensor& out, StreamHandle stream) override;
    void cast(const Tensor& src, Tensor& dst, StreamHandle stream) override;
    void cat(const std::vector<Tensor>& inputs, Tensor& out,
             int64_t dim, StreamHandle stream) override;

    void graph_begin_capture(StreamHandle stream) override;
    void graph_end_capture(StreamHandle stream)   override;
    void graph_launch(StreamHandle stream)        override;

    // ── CUDA-specific helpers (used by model implementations) ─────────────
    int            device_index()   const { return device_index_; }
    cublasHandle_t cublas_handle()  const { return cublas_; }
    CUDAMemoryPool& pool()                { return pool_; }

private:
    // Convert IBackend::StreamHandle (void*) to cudaStream_t.
    static cudaStream_t to_stream(StreamHandle h) {
        return reinterpret_cast<cudaStream_t>(h);
    }

    // Validate that a Tensor lives on this backend's device.
    void check_device(const Tensor& t, const char* arg_name) const;

    // Route GEMM to cuBLAS with correct precision for the tensor dtype.
    void gemm_impl(const Tensor& A, const Tensor& B, Tensor& C,
                   float alpha, float beta,
                   bool trans_a, bool trans_b,
                   cudaStream_t stream);

    int             device_index_;
    cublasHandle_t  cublas_ = nullptr;
    CUDAMemoryPool  pool_;

    // One CUDAGraph per stream for graph capture replay.
    std::unordered_map<cudaStream_t, CUDAGraph> graphs_;
    std::mutex graphs_mu_;
};

// Register the CUDA backend into BackendRegistry at static-init time.
struct CUDABackendRegistrar {
    CUDABackendRegistrar();
};

}  // namespace cuda
}  // namespace sf
