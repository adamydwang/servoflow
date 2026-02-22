cmake_minimum_required(VERSION 3.22)

add_executable(run_inference run_inference.cpp)
target_link_libraries(run_inference PRIVATE servoflow)
target_include_directories(run_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
