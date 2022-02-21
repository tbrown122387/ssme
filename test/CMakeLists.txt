cmake_minimum_required(VERSION 3.12)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -g")

file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/test/*.cpp)
add_executable(${PROJECT_NAME}_test ${SOURCES})


target_include_directories(${PROJECT_NAME}_test 
    PUBLIC 
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_compile_features(${PROJECT_NAME}_test PUBLIC cxx_std_17)
target_link_libraries(${PROJECT_NAME}_test Eigen3::Eigen Catch2::Catch2WithMain Threads::Threads pf::pf)
