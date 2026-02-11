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
file(COPY ${PROJECT_SOURCE_DIR}/data/ 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data
)

# ------------------------------------------------------------
# Tests unitaires
# ------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    
    add_executable(test_fc tests/test_fc.cpp src/layers/fc.cpp)
    target_link_libraries(test_fc PRIVATE HEAAN2::HEAAN2)
    target_include_directories(test_fc PRIVATE ${PROJECT_SOURCE_DIR}/include)
    add_test(NAME test_fc COMMAND test_fc)

    add_executable(test_conv2d tests/test_conv2d.cpp 
        src/layers/conv2d.cpp 
        src/utils/packing.cpp 
        src/utils/key_utils.cpp
        src/utils/io_utils.cpp
    )
    target_link_libraries(test_conv2d PRIVATE HEAAN2::HEAAN2)
    target_include_directories(test_conv2d PRIVATE ${PROJECT_SOURCE_DIR}/include)
    add_test(NAME test_conv2d COMMAND test_conv2d)

    # Test Pooling
    add_executable(test_pooling tests/test_pooling.cpp 
        src/layers/pooling.cpp 
        src/utils/packing.cpp 
        src/utils/key_utils.cpp
    )
    target_link_libraries(test_pooling PRIVATE HEAAN2::HEAAN2)
    target_include_directories(test_pooling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_pooling COMMAND test_pooling)

    # Test ReLU
    add_executable(test_relu tests/test_relu.cpp 
        src/layers/relu.cpp 
        src/utils/packing.cpp 
        src/utils/key_utils.cpp
    )
    target_link_libraries(test_relu PRIVATE HEAAN2::HEAAN2)
    target_include_directories(test_relu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_relu COMMAND test_relu)

    # Test Bootstrapping
    add_executable(test_bootstrap tests/test_bootstrap.cpp 
        src/layers/bootstrapping.cpp 
        src/utils/key_utils.cpp
    )
    target_link_libraries(test_bootstrap PRIVATE HEAAN2::HEAAN2)
    target_include_directories(test_bootstrap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_bootstrap COMMAND test_bootstrap)

    # Test one-hot
    add_executable(test_onehot tests/test_onehot.cpp 
        src/layers/onehot.cpp
        src/layers/bootstrapping.cpp
        src/utils/packing.cpp
        src/utils/key_utils.cpp
    )
    target_link_libraries(test_onehot PRIVATE HEAAN2::HEAAN2)
    target_include_directories(test_onehot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    add_test(NAME test_onehot COMMAND test_onehot)

    if(USE_CUDA)
        target_link_libraries(test_fc PRIVATE CUDA::cudart_static)
    endif()
endif()

message(STATUS "✅ FHE CNN MNIST configuré")
message(STATUS "  HEAAN2: ${HEAAN2_ROOT}")
message(STATUS "  CUDA: ${USE_CUDA}")