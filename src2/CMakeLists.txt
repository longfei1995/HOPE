cmake_minimum_required(VERSION 3.18)
project(APAPlanner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── 自动检测 LibTorch 路径 ─────────────────────────────────────────
# 优先使用 Python 环境中已安装的 PyTorch（CUDA 版需同时有 nvcc）
# 回退顺序：conda HOPE 环境 → 任意激活的 conda 环境 → 本地内嵌 libtorch/
#
# 强制指定路径（跳过自动检测）：
#   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
if(NOT CMAKE_PREFIX_PATH)
    # 1. 查找 Python（优先 conda HOPE 环境）
    find_program(PYTHON_EXEC
        NAMES python3 python
        HINTS
            /home/hyh/miniconda3/envs/HOPE/bin
            ENV CONDA_PREFIX
        NO_CMAKE_SYSTEM_PATH
    )

    # 2. 通过 Python 获取 PyTorch CMake 路径
    if(PYTHON_EXEC)
        execute_process(
            COMMAND ${PYTHON_EXEC} -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE _TORCH_CMAKE_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _TORCH_QUERY_RESULT
        )
    endif()

    # 3. PyTorch CUDA 版需要 nvcc；若无 nvcc 则回退内嵌 CPU 版
    if(_TORCH_CMAKE_PREFIX AND _TORCH_QUERY_RESULT EQUAL 0)
        find_program(NVCC_EXEC nvcc
            HINTS
                /home/hyh/miniconda3/envs/HOPE/bin
                ENV CUDA_PATH
                /usr/local/cuda/bin
            NO_CMAKE_SYSTEM_PATH
        )
        if(NVCC_EXEC)
            message(STATUS "[LibTorch] Python 环境 (CUDA): ${PYTHON_EXEC}")
            message(STATUS "[LibTorch] nvcc: ${NVCC_EXEC}")
            list(APPEND CMAKE_PREFIX_PATH ${_TORCH_CMAKE_PREFIX})
        else()
            message(STATUS "[LibTorch] Python PyTorch 为 CUDA 版但未找到 nvcc")
            message(STATUS "[LibTorch] 安装 nvcc：conda install -c nvidia/label/cuda-12.1.0 cuda-nvcc -n HOPE")
            message(STATUS "[LibTorch] 回退：使用内嵌 libtorch/ (CPU-only)")
            list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)
        endif()
    else()
        message(STATUS "[LibTorch] 未检测到 Python PyTorch，使用内嵌 libtorch/")
        list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)
    endif()
endif()

find_package(Torch REQUIRED)

# ─── APAPlanner 静态库 ─────────────────────────────────────────────
add_library(apa_planner STATIC
    src/apa_planner.cpp
)
target_include_directories(apa_planner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(apa_planner PUBLIC
    ${TORCH_LIBRARIES}
)

# ─── 可执行演示程序 ────────────────────────────────────────────────
add_executable(apa_demo
    src/main.cpp
)
target_link_libraries(apa_demo PRIVATE
    apa_planner
)

# Linux 上链接 pthread
if(UNIX AND NOT APPLE)
    target_link_libraries(apa_demo PRIVATE pthread)
endif()

# Windows 上拷贝 LibTorch DLL
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET apa_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS} $<TARGET_FILE_DIR:apa_demo>
    )
endif()
