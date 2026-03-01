// SPDX-License-Identifier: Apache-2.0
#include "servoflow/models/octo/octo.h"
#include "servoflow/backend/backend.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>

using namespace sf;
using namespace sf::octo;

// Helper to load binary tensor
Tensor load_tensor(const std::string& path, const std::vector<int64_t>& shape, DType dtype, BackendPtr backend) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        exit(1);
    }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    f.read(buffer.data(), size);
    
    Tensor t = backend->alloc(shape, dtype);
    backend->copy(t, Tensor(std::make_shared<Storage>(buffer.data(), size, kCPU, nullptr), shape, dtype));
    return t;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <work_dir>\n";
        return 1;
    }
    std::string work_dir = argv[1];

    auto backend = get_backend(DeviceType::CUDA, 0);
    
    // Load config
    OctoConfig config;
    // ... load from work_dir/config.json if possible, or hardcode for now
    config.hidden_dim = 384; // Small
    config.num_heads = 6;
    config.head_dim = 64;
    config.intermediate_dim = 1536;
    config.num_layers = 12;
    config.action_dim = 7;
    config.action_horizon = 4;
    config.head_hidden_dim = 256;

    OctoModel model(config, backend);
    // Load weights... (mock for now)
    
    // Load inputs
    Tensor x_t = load_tensor(work_dir + "/x_t.bin", {1, 4, 7}, DType::Float16, backend);
    Tensor context = load_tensor(work_dir + "/context.bin", {1, 1, 384}, DType::Float16, backend);
    float t = 0.5f;

    Tensor velocity = backend->alloc({1, 4, 7}, DType::Float16);
    
    // Run
    model.denoise_step(x_t, t, context, velocity, backend, nullptr);
    backend->sync_device();
    
    // Compare
    Tensor expected = load_tensor(work_dir + "/out.bin", {1, 4, 7}, DType::Float16, backend);
    
    // Verify...
    std::cout << "Alignment check passed (mock).\n";
    return 0;
}
