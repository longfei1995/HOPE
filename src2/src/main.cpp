#include "apa_planner.h"
#include "bicycle_model.h"
#include "scene_loader.h"
#include "visualizer.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ═══════════════════════════════════════════════════════════════════
// 场景可视化模式：加载 JSON → 运动学传播 → 绘制场景和曲线
// ═══════════════════════════════════════════════════════════════════
static int runSceneMode(const std::string& json_path,
                        const std::string& scene_out,
                        const std::string& traj_out) {
    std::cout << "加载场景: " << json_path << std::endl;
    SceneData scene = loadScene(json_path);
    std::cout << "  状态: " << scene.status
              << "  步数: " << scene.total_steps
              << "  障碍物: " << scene.obstacles.size() << std::endl;

    // 构建运动学模型
    BicycleModel model(scene.vehicle_params);

    // 从起点开始传播
    std::vector<ControlInput> controls;
    controls.reserve(scene.steps.size());
    for (const auto& s : scene.steps) {
        controls.push_back({s.actual_steer, s.actual_speed});
    }

    auto trajectory = model.propagate(
        scene.start_x, scene.start_y, scene.start_yaw, controls);

    // 打印轨迹表格
    std::cout << "\n";
    std::cout << "===================================================================\n";
    std::cout << "  步骤  |     x(m)    |     y(m)    |   yaw(rad)  | kappa(1/m) |  v(m/s)\n";
    std::cout << "--------+-------------+-------------+-------------+------------+--------\n";

    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& pt = trajectory[i];
        printf("  %5zu | %+11.4f | %+11.4f | %+11.4f | %+10.4f | %+7.2f\n",
               i, pt.x, pt.y, pt.yaw, pt.kappa, pt.v);
    }
    std::cout << "===================================================================\n";

    // 验证与 Python 端的一致性
    if (!scene.steps.empty()) {
        const auto& last_step = scene.steps.back();
        const auto& last_pt   = trajectory.back();
        float dx = last_pt.x - last_step.x;
        float dy = last_pt.y - last_step.y;
        float err = std::sqrt(dx * dx + dy * dy);
        printf("\n轨迹终点误差 (C++ vs Python): %.6f m\n", err);
        if (err > 0.01f) {
            std::cerr << "警告: 终点误差较大，请检查运动学参数是否一致\n";
        }
    }

    // 绘制可视化图
    Visualizer vis(1200, 40.0f);

    // 以起点（后轴中心）为原点
    vis.drawScene(scene, trajectory, scene.start_x, scene.start_y, scene_out);
    std::cout << "\n场景图已保存: " << scene_out << std::endl;

    vis.drawTrajectoryPlots(trajectory, traj_out);
    std::cout << "轨迹曲线已保存: " << traj_out << std::endl;

    return 0;
}

// ═══════════════════════════════════════════════════════════════════
// Fake 数据推理模式（原有逻辑）
// ═══════════════════════════════════════════════════════════════════

std::vector<float> generateFakeLidar(std::mt19937& rng) {
    std::vector<float> lidar(120);
    for (int i = 0; i < 120; ++i) {
        float angle = static_cast<float>(i) / 120.0f * 2.0f * static_cast<float>(M_PI);
        float base_dist = 5.0f + 3.0f * std::sin(angle);
        if (angle > 4.0f && angle < 5.5f) {
            base_dist = 1.5f + 0.5f * std::abs(angle - 4.7f);
        }
        std::normal_distribution<float> noise(0.0f, 0.1f);
        lidar[i] = std::max(0.0f, std::min(10.0f, base_dist + noise(rng)));
    }
    return lidar;
}

std::vector<float> generateFakeTarget() {
    float x     = -3.0f;
    float y     = -2.5f;
    float theta = static_cast<float>(M_PI) / 2.0f;
    return {x, y, theta, std::cos(theta), std::sin(theta)};
}

std::vector<float> generateFakeImage(std::mt19937& rng) {
    std::vector<float> img(3 * 64 * 64);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : img) {
        v = dist(rng);
    }
    return img;
}

std::vector<float> generateFakeActionMask(std::mt19937& /*rng*/) {
    std::vector<float> mask(42, 1.0f);
    for (int i = 15; i <= 20; ++i) {
        mask[i] = 0.0f;
    }
    for (int i = 21; i <= 25; ++i) {
        mask[i] = 0.0f;
    }
    return mask;
}

static int runInferenceMode(const std::string& model_path, int num_steps) {
    std::cout << "加载模型: " << model_path << std::endl;
    APAPlanner planner(model_path, /*deterministic=*/true);
    std::cout << "模型加载成功！\n" << std::endl;

    std::mt19937 rng(42);

    // 预热
    {
        auto lidar = generateFakeLidar(rng);
        auto target = generateFakeTarget();
        auto img = generateFakeImage(rng);
        auto mask = generateFakeActionMask(rng);
        planner.run(lidar, target, img, mask);
        std::cout << "预热完成\n" << std::endl;
    }

    std::cout << "===========================================================\n";
    std::cout << " 步骤 | 转向角(rad) | 速度(m/s) | 动作索引 | 耗时(ms)\n";
    std::cout << "------+-------------+-----------+----------+---------\n";

    double total_time = 0.0;
    for (int step = 0; step < num_steps; ++step) {
        auto lidar = generateFakeLidar(rng);
        auto target = generateFakeTarget();
        auto img = generateFakeImage(rng);
        auto mask = generateFakeActionMask(rng);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = planner.run(lidar, target, img, mask);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_time += ms;

        printf(" %4d | %+11.4f | %+9.2f |   %4d   | %7.2f\n",
               step, result.actual_steer, result.actual_speed,
               result.action_idx, ms);
    }

    std::cout << "===========================================================\n";
    printf("平均推理时间: %.2f ms / 步\n", total_time / num_steps);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法:\n";
        std::cerr << "  推理模式: " << argv[0] << " <模型路径> [推理次数]\n";
        std::cerr << "  场景模式: " << argv[0] << " --scene <json路径> [--scene-out scene.png] [--traj-out trajectory.png]\n";
        std::cerr << "\n示例:\n";
        std::cerr << "  " << argv[0] << " ./model/actor_traced.pt 10\n";
        std::cerr << "  " << argv[0] << " --scene ../data/episode.json\n";
        return 1;
    }

    // 场景可视化模式
    if (std::string(argv[1]) == "--scene") {
        if (argc < 3) {
            std::cerr << "错误: --scene 后需要指定 JSON 文件路径\n";
            return 1;
        }
        std::string json_path = argv[2];
        std::string scene_out = "scene.png";
        std::string traj_out  = "trajectory.png";

        // 解析可选参数
        for (int i = 3; i < argc - 1; ++i) {
            if (std::string(argv[i]) == "--scene-out") {
                scene_out = argv[++i];
            } else if (std::string(argv[i]) == "--traj-out") {
                traj_out = argv[++i];
            }
        }

        return runSceneMode(json_path, scene_out, traj_out);
    }

    // 推理模式（原有逻辑）
    std::string model_path = argv[1];
    int num_steps = (argc >= 3) ? std::atoi(argv[2]) : 10;
    return runInferenceMode(model_path, num_steps);
}
