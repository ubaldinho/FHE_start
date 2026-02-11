# fhe_cnn_mnist/CMakeLists.txt
cmake_minimum_required(VERSION 3.23)
project(FHE_CNN_MNIST VERSION 1.0)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
option(USE_CUDA "Utiliser CUDA pour accélération GPU" OFF)
option(BUILD_TESTS "Compiler les tests unitaires" ON)

# ------------------------------------------------------------
# C++ standard
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# ------------------------------------------------------------
# Trouver HEAAN2 (dans devkit)
# ------------------------------------------------------------
set(HEAAN2_ROOT "/home/user/devkit" CACHE PATH "Path to HEAAN2")
find_package(HEAAN2 REQUIRED HINTS ${HEAAN2_ROOT})

# ------------------------------------------------------------
# Includes - CRITIQUE : include/ et HEAAN2
# ------------------------------------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/include 
    ${PROJECT_SOURCE_DIR}/include/fhe_cnn   
    ${PROJECT_SOURCE_DIR}/src    
    ${HEAAN2_INCLUDE_DIRS}                   
)

# ------------------------------------------------------------
# CUDA
# ------------------------------------------------------------
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DHEAAN_EXAMPLES_CUDA)
endif()

# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------


file(GLOB_RECURSE CNN_SOURCES
    src/*.cpp
    src/layers/*.cpp
    src/utils/*.cpp
)

# ------------------------------------------------------------
# Exécutable principal
# ------------------------------------------------------------
add_executable(cnn_mnist ${CNN_SOURCES})
target_link_libraries(cnn_mnist PRIVATE HEAAN2::HEAAN2)

if(USE_CUDA)
    target_link_libraries(cnn_mnist PRIVATE CUDA::cudart_static)
endif()

# ------------------------------------------------------------
# Copier les données dans le dossier de build
# ------------------------------------------------------------
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data
)

# ------------------------------------------------------------
# Tests unitaires
# ------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_fc tests/test_fc.cpp src/layers/fc.cpp)
    target_link_libraries(test_fc PRIVATE HEAAN2::HEAAN2)
    target_include_directories(test_fc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_fc COMMAND test_fc)
    
    if(USE_CUDA)
        target_link_libraries(test_fc PRIVATE CUDA::cudart_static)
    endif()
endif()

message(STATUS "✅ FHE CNN MNIST configuré")
message(STATUS "  HEAAN2: ${HEAAN2_ROOT}")
message(STATUS "  CUDA: ${USE_CUDA}")