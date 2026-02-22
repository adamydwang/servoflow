cmake_minimum_required(VERSION 3.22)

# ── Unit tests ────────────────────────────────────────────────────────────────
function(sf_add_test name)
    add_executable(${name} unit/${name}.cpp)
    target_link_libraries(${name} PRIVATE servoflow GTest::gtest_main)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    gtest_discover_tests(${name})
endfunction()

include(GoogleTest)

sf_add_test(test_dtype)
sf_add_test(test_shape)
sf_add_test(test_tensor)
sf_add_test(test_sampler)

# Integration tests require CUDA.
if(SF_ENABLE_CUDA)
    add_executable(test_vla_pipeline integration/test_vla_pipeline.cpp)
    target_link_libraries(test_vla_pipeline PRIVATE servoflow GTest::gtest_main)
    gtest_discover_tests(test_vla_pipeline)
endif()
