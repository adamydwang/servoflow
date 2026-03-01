// SPDX-License-Identifier: Apache-2.0
// ServoFlow — Octo Inference Benchmark
//
// Demonstrates how to load Octo and run inference with detailed performance profiling.
//
// Usage:
//   ./octo_inference <checkpoint_dir> [num_steps]

#include "servoflow/engine/inference_engine.h"
#include "servoflow/models/octo/octo.h"
#include "servoflow/sampling/sampler.h"
#include "servoflow/backend/cuda/cuda_backend.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace sf;
using namespace sf::octo;

// Helper for formatted time printing
std::string format_duration(double ms) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << ms << " ms";
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <checkpoint_dir> [num_steps]\n";
        // return 1; // Allow running without args for mock
    }
    std::string checkpoint_dir = (argc > 1) ? argv[1] : "mock_ckpt";
    int num_steps = (argc > 2) ? std::atoi(argv[2]) : 50;

    // 1. Initialize Backend (CUDA device 0)
    auto backend = std::make_shared<cuda::CUDABackend>(0);
    std::cout << "======================================================================\n";
    std::cout << "  ServoFlow Octo Benchmark\n";
    std::cout << "======================================================================\n";
    std::cout << "  Backend    : " << backend->caps().device_name << "\n";

    // 2. Configure Engine
    EngineConfig config;
    config.device = Device(DeviceType::CUDA, 0);
    config.compute_dtype = DType::Float16; // FP16 for speed
    config.num_denoise_steps = 10;        // Standard for Octo
    config.use_cuda_graph = true;         // Enable CUDA Graph
    config.pinned_output = true;          // Fast D2H transfer
    config.cache_condition = true;        // Enable condition caching

    // 3. Load Model (Mock Config for Benchmark)
    OctoConfig octo_cfg;
    octo_cfg.hidden_dim = 384;      // Small
    octo_cfg.num_heads = 6;
    octo_cfg.head_dim = 64;
    octo_cfg.intermediate_dim = 1536;
    octo_cfg.num_layers = 12;
    octo_cfg.action_dim = 7;
    octo_cfg.action_horizon = 4;
    octo_cfg.head_hidden_dim = 256;
    octo_cfg.compute_dtype = DType::Float16;

    std::cout << "  Model      : Octo-Small (Hidden: " << octo_cfg.hidden_dim << ", Layers: " << octo_cfg.num_layers << ")\n";

    auto model = std::make_shared<OctoModel>(octo_cfg, backend);
    // model->load_weights(checkpoint_dir); // Skip for benchmark if just testing compute

    // 4. Create Sampler (DDIM/DDPM)
    // Octo typically uses DDIM.
    auto sampler = std::make_shared<FlowMatchingSampler>(/*use_cuda_graph=*/true); // Placeholder, implement DDIM later

    // 5. Create Inference Engine
    InferenceEngine engine(model, sampler, config);

    // 6. Prepare Inputs
    // Mock inputs
    VLAInput input;
    // ... setup inputs
    
    // 7. Warmup
    std::cout << "  Warmup...\n";
    // engine.infer(input); 
    // Need proper inputs for infer()
    
    // Benchmark Encoder
    std::cout << "  Benchmarking Encoder...\n";
    VLAInput vla_input; // Dummy
    auto t_enc_start = std::chrono::high_resolution_clock::now();
    Tensor context_enc = model->encode_condition(vla_input, backend, nullptr);
    backend->sync_device();
    auto t_enc_end = std::chrono::high_resolution_clock::now();
    double enc_ms = std::chrono::duration<double, std::milli>(t_enc_end - t_enc_start).count();
    std::cout << "  Encoder Latency: " << format_duration(enc_ms) << "\n";
    
    // Since we don't have proper inputs setup in this mock, we skip engine.infer call 
    // and just benchmark model.denoise_step directly for raw compute speed.
    
    Tensor x_t = backend->alloc({1, octo_cfg.action_horizon, octo_cfg.action_dim}, DType::Float16);
    // Use context from encoder
    Tensor context = context_enc;
    Tensor velocity = backend->alloc({1, octo_cfg.action_horizon, octo_cfg.action_dim}, DType::Float16);
    float t = 0.5f;
    
    backend->fill(x_t, 0.5f);
    // Context already filled by encoder
    
    // Benchmark Loop
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "  Running " << num_steps << " denoise steps (Raw Compute)...\n";
    std::cout << "----------------------------------------------------------------------\n";
    
    std::vector<double> latencies;
    latencies.reserve(num_steps);
    
    for (int i = 0; i < num_steps; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();
        
        model->denoise_step(x_t, t, context, velocity, backend, nullptr);
        backend->sync_device();
        
        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        latencies.push_back(ms);
        
        if (i % 10 == 0) std::cout << "  Step " << i << ": " << format_duration(ms) << "\n";
    }
    
    // Stats
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    std::cout << "  Avg Latency: " << format_duration(avg) << "\n";
    std::cout << "  Throughput : " << (1000.0 / avg) << " steps/sec\n";
    
    return 0;
}
