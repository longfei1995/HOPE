cmake_minimum_required(VERSION 3.18)
project(APAPlanner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# ─── APAPlanner 静态库 ─────────────────────────────────────────────
add_library(apa_planner STATIC
    src/apa_planner.cpp
    src/visualizer.cpp
)
target_include_directories(apa_planner PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(apa_planner PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# ─── 可执行演示程序 ────────────────────────────────────────────────
add_executable(apa_demo
    src/main.cpp
)
target_link_libraries(apa_demo PRIVATE
    apa_planner
    pthread
)



