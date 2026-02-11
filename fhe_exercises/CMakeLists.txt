cmake_minimum_required(VERSION 3.23)
project(
    FHE_Exercices
    VERSION 1.0
    DESCRIPTION "Exercices TP FHE - HEAAN2"
    LANGUAGES CXX
)

# Option pour CUDA (optionnel)
option(USE_CUDA "Utiliser CUDA pour l'accélération GPU" OFF)

# Trouver HEAAN2 (devkit est au même niveau)
set(HEAAN2_ROOT "../devkit")
find_package(HEAAN2 REQUIRED HINTS ${HEAAN2_ROOT})

# Configuration CUDA
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DHEAAN_EXAMPLES_CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Liste des exécutables (un par exercice)
set(EXERCICES
    horner
    goldschmidt
    rotate_sum
    row_method
    diagonal_method
    diagonal_bsgs
    paterson_stockmeyer
)

# Pour chaque exercice, créer un exécutable
foreach(ex ${EXERCICES})
    # Vérifier si le fichier source existe
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${ex}.cpp")
        add_executable(${ex} src/${ex}.cpp)
        target_link_libraries(${ex} PRIVATE HEAAN2::HEAAN2)
        
        if(USE_CUDA)
            target_link_libraries(${ex} PRIVATE CUDA::cudart_static)
        endif()
        
        # Options de compilation
        target_compile_options(${ex} PRIVATE -Wall -Wextra -O3)
        
        message(STATUS "Ajout de l'exercice: ${ex}")
    else()
        message(WARNING "Fichier src/${ex}.cpp non trouvé - ignoré")
    endif()
endforeach()

# Option pour tout compiler en une commande
add_custom_target(all_exercices DEPENDS ${EXERCICES})