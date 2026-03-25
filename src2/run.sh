#!/bin/bash

# 泊车仿真运行脚本
# 支持构建和运行 C++ 泊车仿真程序

# 颜色定义
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认值
DEFAULT_JSON_PATH="${SCRIPT_DIR}/data/episode.json"
DEFAULT_SCENE_OUT="scene.png"
DEFAULT_TRAJ_OUT="trajectory.png"
DEFAULT_MODEL_PATH="${SCRIPT_DIR}/model/actor_traced.pt"
DEFAULT_INFERENCE_STEPS=10

# 显示使用说明
show_usage() {
    echo -e "${GREEN}泊车仿真运行脚本${NC}"
    echo -e ""
    echo -e "用法: ./run.sh <命令> [参数]"
    echo -e ""
    echo -e "命令:"
    echo -e "  build        构建 C++ 项目"
    echo -e "  scene        运行场景可视化模式"
    echo -e "  inference    运行推理模式（可选）"
    echo -e "  help         显示此帮助信息"
    echo -e ""
    echo -e "场景模式参数:"
    echo -e "  --json <路径>      指定 JSON 场景文件路径"
    echo -e "  --scene-out <路径> 指定场景图像输出路径"
    echo -e "  --traj-out <路径>  指定轨迹图像输出路径"
    echo -e ""
    echo -e "推理模式参数:"
    echo -e "  <模型路径>         指定模型文件路径"
    echo -e "  <推理次数>         指定推理次数（默认 10）"
    echo -e ""
    echo -e "示例:"
    echo -e "  ./run.sh build"
    echo -e "  ./run.sh scene"
    echo -e "  ./run.sh scene --json ./data/episode.json"
    echo -e "  ./run.sh inference ./model/actor_traced.pt 10"
}

# 检查依赖项
check_dependencies() {
    echo -e "${YELLOW}检查依赖项...${NC}"
    
    # 检查 CMake
    if ! command -v cmake &> /dev/null; then
        echo -e "${RED}错误: CMake 未安装${NC}"
        echo -e "请运行: sudo apt install cmake"
        return 1
    fi
    
    # 检查 g++
    if ! command -v g++ &> /dev/null; then
        echo -e "${RED}错误: g++ 未安装${NC}"
        echo -e "请运行: sudo apt install g++"
        return 1
    fi
    
    # 检查 libtorch
    if [ ! -d "/opt/libtorch" ]; then
        echo -e "${YELLOW}警告: libtorch 未在 /opt/libtorch 找到${NC}"
        echo -e "请确保 libtorch 已安装并正确配置"
    fi
    
    # 检查 OpenCV
    if ! pkg-config --exists opencv4; then
        echo -e "${YELLOW}警告: OpenCV 未找到${NC}"
        echo -e "请运行: sudo apt install libopencv-dev"
    fi
    
    echo -e "${GREEN}依赖项检查完成${NC}"
    return 0
}

# 构建项目
build_project() {
    echo -e "${YELLOW}构建项目...${NC}"
    
    # 创建 build 目录
    mkdir -p "${SCRIPT_DIR}/build"
    cd "${SCRIPT_DIR}/build"
    
    # 运行 CMake
    echo -e "${YELLOW}运行 CMake...${NC}"
    cmake ..
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: CMake 配置失败${NC}"
        return 1
    fi
    
    # 运行 make
    echo -e "${YELLOW}编译项目...${NC}"
    make -j4
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 编译失败${NC}"
        return 1
    fi
    
    # 检查可执行文件
    if [ ! -f "apa_demo" ]; then
        echo -e "${RED}错误: 可执行文件未生成${NC}"
        return 1
    fi
    
    echo -e "${GREEN}构建成功！${NC}"
    return 0
}

# 运行场景可视化模式
run_scene() {
    local json_path="${DEFAULT_JSON_PATH}"
    local scene_out="${DEFAULT_SCENE_OUT}"
    local traj_out="${DEFAULT_TRAJ_OUT}"
    
    # 解析参数
    shift
    while [ $# -gt 0 ]; do
        case "$1" in
            --json)
                json_path="$2"
                shift 2
                ;;
            --scene-out)
                scene_out="$2"
                shift 2
                ;;
            --traj-out)
                traj_out="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}错误: 未知参数 '$1'${NC}"
                show_usage
                return 1
                ;;
        esac
    done
    
    # 检查可执行文件
    if [ ! -f "${SCRIPT_DIR}/build/apa_demo" ]; then
        echo -e "${YELLOW}可执行文件未找到，正在构建...${NC}"
        build_project
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi
    
    # 检查 JSON 文件
    if [ ! -f "${json_path}" ]; then
        echo -e "${RED}错误: JSON 文件不存在: ${json_path}${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}运行场景可视化模式...${NC}"
    echo -e "JSON 文件: ${json_path}"
    echo -e "场景输出: ${scene_out}"
    echo -e "轨迹输出: ${traj_out}"
    
    "${SCRIPT_DIR}/build/apa_demo" --scene "${json_path}" --scene-out "${scene_out}" --traj-out "${traj_out}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}场景可视化完成！${NC}"
        return 0
    else
        echo -e "${RED}错误: 场景可视化失败${NC}"
        return 1
    fi
}

# 运行推理模式
run_inference() {
    local model_path="${DEFAULT_MODEL_PATH}"
    local num_steps="${DEFAULT_INFERENCE_STEPS}"
    
    # 解析参数
    if [ $# -ge 1 ]; then
        model_path="$1"
    fi
    if [ $# -ge 2 ]; then
        num_steps="$2"
    fi
    
    # 检查可执行文件
    if [ ! -f "${SCRIPT_DIR}/build/apa_demo" ]; then
        echo -e "${YELLOW}可执行文件未找到，正在构建...${NC}"
        build_project
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi
    
    # 检查模型文件
    if [ ! -f "${model_path}" ]; then
        echo -e "${RED}错误: 模型文件不存在: ${model_path}${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}运行推理模式...${NC}"
    echo -e "模型文件: ${model_path}"
    echo -e "推理次数: ${num_steps}"
    
    "${SCRIPT_DIR}/build/apa_demo" "${model_path}" "${num_steps}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}推理完成！${NC}"
        return 0
    else
        echo -e "${RED}错误: 推理失败${NC}"
        return 1
    fi
}

# 主函数
main() {
    # 检查依赖项
    check_dependencies
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # 解析命令
    if [ $# -eq 0 ]; then
        show_usage
        return 0
    fi
    
    case "$1" in
        build)
            build_project
            ;;
        scene)
            run_scene "$@"
            ;;
        inference)
            shift
            run_inference "$@"
            ;;
        help)
            show_usage
            ;;
        *)
            echo -e "${RED}错误: 未知命令 '$1'${NC}"
            show_usage
            return 1
            ;;
    esac
}

# 执行主函数
main "$@"