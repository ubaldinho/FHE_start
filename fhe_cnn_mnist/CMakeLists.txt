cmake_minimum_required(VERSION 3.23)
project(
    FHE_CNN_MNIST
    VERSION 1.0
    DESCRIPTION "CNN homomorphe pour MNIST - Projet TP"
    LANGUAGES CXX
)

# Option pour CUDA (fortement recommandé pour le CNN)
option(USE_CUDA "Utiliser CUDA pour l'accélération GPU" ON)
option(WITH_BOOTSTRAP "Activer le bootstrapping pour circuits profonds" ON)

# Trouver HEAAN2
set(HEAAN2_ROOT "../devkit")
find_package(HEAAN2 REQUIRED HINTS ${HEAAN2_ROOT})

# Configuration CUDA
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DHEAAN_EXAMPLES_CUDA)
    message(STATUS "Compilation avec CUDA activé")
else()
    message(WARNING "CUDA désactivé - le CNN sera très lent sur CPU")
endif()

# Bootstrap
if(WITH_BOOTSTRAP)
    add_definitions(-DWITH_BOOTSTRAP)
    message(STATUS "Bootstrapping activé")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fichiers sources du CNN
set(CNN_SOURCES
    src/cnn_inference.cpp
    src/layers/conv2d.cpp
    src/layers/pooling.cpp
    src/layers/relu.cpp
    src/layers/fully_connected.cpp
    src/layers/bootstrapping.cpp
    src/utils/io_utils.cpp
    src/utils/measurements.cpp
)

# Vérifier l'existence des fichiers
foreach(file ${CNN_SOURCES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${file}")
        message(WARNING "Fichier ${file} non trouvé")
    endif()
endforeach()

# Exécutable principal
add_executable(cnn_mnist ${CNN_SOURCES})

# Lier HEAAN2
target_link_libraries(cnn_mnist PRIVATE HEAAN2::HEAAN2)

# Lier CUDA si activé
if(USE_CUDA)
    target_link_libraries(cnn_mnist PRIVATE CUDA::cudart_static)
endif()

# Inclure les répertoires
target_include_directories(cnn_mnist PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

# Options de compilation agressives pour la performance
target_compile_options(cnn_mnist PRIVATE
    -Wall
    -Wextra
    -O3
    -march=native
    -ffast-math
)

# Copier les données (MNIST, poids) dans le dossier de build
file(COPY data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

message(STATUS "Projet CNN MNIST configuré")
message(STATUS "  - CUDA: ${USE_CUDA}")
message(STATUS "  - Bootstrap: ${WITH_BOOTSTRAP}")
message(STATUS "  - Données: ${CMAKE_CURRENT_BINARY_DIR}/data")