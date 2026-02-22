cmake_minimum_required(VERSION 3.22)

if(SF_ENABLE_CUDA)
    add_executable(bench_attention bench_attention.cu)
    target_link_libraries(bench_attention PRIVATE servoflow CUDA::cudart)
    target_include_directories(bench_attention PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include)

    add_executable(bench_pipeline bench_pipeline.cpp)
    target_link_libraries(bench_pipeline PRIVATE servoflow)
    target_include_directories(bench_pipeline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include)
endif()
