cmake_minimum_required(VERSION 3.20)
project(BlackholeSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# Qt6
find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGL OpenGLWidgets)

# CUDA
find_package(CUDAToolkit REQUIRED)
# CUDA 13.x dropped Pascal (sm_60/61). sm_120 requires CUDA 12.8+.
# sm_75:  GTX 16xx / RTX 20xx (Turing)      <- CUDA 13.x minimum
# sm_80:  A100 (Ampere datacenter)
# sm_86:  RTX 30xx (Ampere gaming)
# sm_89:  RTX 40xx (Ada Lovelace)
# sm_90:  H100 / H200 (Hopper)
# sm_100: B100 / B200 (Blackwell datacenter)
# sm_120: RTX 50xx (Blackwell consumer)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;100;120")

# vcpkg packages
find_package(glm CONFIG REQUIRED)

# Collect sources
set(CXX_SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/BlackHoleWidget.cpp
    src/TextureLoader.cpp
    src/stb_impl.cpp
    src/GeodesicDebugWidget.cpp
    src/CpuRenderer.cpp
)

set(CUDA_SOURCES
    src/cuda/RayTracer.cu
)

set(HEADERS
    src/MainWindow.h
    src/BlackHoleWidget.h
    src/TextureLoader.h
    src/Camera.h
    src/physics/KerrNewman.h
    src/cuda/RayTracer.cuh
    src/GeodesicDebugWidget.h
)

qt6_add_resources(QRC_SOURCES resources/resources.qrc)

add_executable(BlackholeSim
    ${CXX_SOURCES}
    ${CUDA_SOURCES}
    ${HEADERS}
    ${QRC_SOURCES}
)

set_target_properties(BlackholeSim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(BlackholeSim PRIVATE
    src
    src/physics
    src/cuda
    ${CMAKE_SOURCE_DIR}/vcpkg_installed/x64-windows/include
)

target_link_libraries(BlackholeSim PRIVATE
    Qt6::Core
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    CUDA::cudart
    CUDA::cuda_driver
    glm::glm
)

# Copy background textures next to executable, if present locally.
# These are large NASA-sourced assets kept out of git (size + redistribution
# license TBD — see assets/textures/README.md), so a fresh clone may not have
# them; only wire up copy commands for files that actually exist at configure
# time so the build doesn't fail when they're missing (app falls back to a
# procedural starfield in that case).
set(BH_TEXTURE_DIR "${CMAKE_SOURCE_DIR}/assets/textures")
set(BH_TEXTURE_FILES
    starmap_2020_4k_gal_print.jpg
    starmap_2020_4k_gal.exr
    starmap_2020_8k_gal.png
    starmap_2020_8k_gal.exr
)
foreach(_bh_tex ${BH_TEXTURE_FILES})
    if(EXISTS "${BH_TEXTURE_DIR}/${_bh_tex}")
        add_custom_command(TARGET BlackholeSim POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${BH_TEXTURE_DIR}/${_bh_tex}"
                $<TARGET_FILE_DIR:BlackholeSim>
        )
    endif()
endforeach()
