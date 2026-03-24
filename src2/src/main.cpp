#include "apa_planner.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * 生成模拟的激光雷达数据（120 个射线的距离值）
 * 模拟前方有障碍物、右侧有车位壁的典型泊车场景
 */
std::vector<float> generateFakeLidar(std::mt19937& rng) {
    std::vector<float> lidar(120);
    for (int i = 0; i < 120; ++i) {
        float angle = static_cast<float>(i) / 120.0f * 2.0f * static_cast<float>(M_PI);
        // 前方距离较近，模拟前方有车
        float base_dist = 5.0f + 3.0f * std::sin(angle);
        // 右侧（角度 ~270°）距离较近，模拟车位壁
        if (angle > 4.0f && angle < 5.5f) {
            base_dist = 1.5f + 0.5f * std::abs(angle - 4.7f);
        }
        std::normal_distribution<float> noise(0.0f, 0.1f);
        lidar[i] = std::max(0.0f, std::min(10.0f, base_dist + noise(rng)));
    }
    return lidar;
}

/**
 * 生成模拟的目标位姿 [x, y, theta, cos(theta), sin(theta)]
 * 模拟车位在车辆右后方
 */
std::vector<float> generateFakeTarget() {
    float x     = -3.0f;
    float y     = -2.5f;
    float theta = static_cast<float>(M_PI) / 2.0f;
    return {x, y, theta, std::cos(theta), std::sin(theta)};
}

/**
 * 生成模拟的摄像头图像（3×64×64，归一化到 [0,1]）
 */
std::vector<float> generateFakeImage(std::mt19937& rng) {
    std::vector<float> img(3 * 64 * 64);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : img) {
        v = dist(rng);
    }
    return img;
}

/**
 * 生成模拟的动作掩码（42 维，1=可行，0=碰撞）
 * 部分大角度转弯动作设为碰撞
 */
std::vector<float> generateFakeActionMask(std::mt19937& /*rng*/) {
    std::vector<float> mask(42, 1.0f);
    // 模拟：部分前进右转会碰撞
    for (int i = 15; i <= 20; ++i) {
        mask[i] = 0.0f;
    }
    // 模拟：部分后退左转会碰撞
    for (int i = 21; i <= 25; ++i) {
        mask[i] = 0.0f;
    }
    return mask;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <模型路径> [推理次数]\n";
        std::cerr << "示例: " << argv[0] << " ./model/actor_traced.pt 10\n";
        return 1;
    }

    std::string model_path = argv[1];
    int num_steps = (argc >= 3) ? std::atoi(argv[2]) : 10;

    // 创建规划器（确定性模式）
    std::cout << "加载模型: " << model_path << std::endl;
    APAPlanner planner(model_path, /*deterministic=*/true);
    std::cout << "模型加载成功！\n" << std::endl;

    std::mt19937 rng(42);

    // 预热推理（首次推理通常较慢）
    {
        auto lidar = generateFakeLidar(rng);
        auto target = generateFakeTarget();
        auto img = generateFakeImage(rng);
        auto mask = generateFakeActionMask(rng);
        planner.run(lidar, target, img, mask);
        std::cout << "预热完成\n" << std::endl;
    }

    // 多步推理演示
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
