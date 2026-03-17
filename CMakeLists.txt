cmake_minimum_required(VERSION 3.20)

project(FeatureSelector
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 12.1.0
    )
FetchContent_MakeAvailable(fmt)

add_executable(feature_selector
    main.cpp
)

target_include_directories(feature_selector PRIVATE
    include/
)

target_link_libraries(feature_selector PRIVATE
    fmt::fmt
)

target_compile_options(feature_selector PRIVATE
	-O3
	-std=c++20
	-Wall
)
