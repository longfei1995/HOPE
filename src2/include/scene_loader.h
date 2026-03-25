#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bicycle_model.h"
#include <nlohmann/json.hpp>

/**
 * 场景数据：从 Python 端 export_episode.py 导出的 JSON 中加载
 */
struct SceneData {
    /// 多边形（用顶点列表表示）
    using Polygon = std::vector<std::pair<float, float>>;

    // 起点和目标
    float start_x, start_y, start_yaw;
    float dest_x, dest_y, dest_yaw;

    Polygon start_box;       ///< 起始车辆包围盒
    Polygon parking_lot;     ///< 目标车位多边形

    std::vector<Polygon> obstacles;  ///< 所有障碍物

    /// 每步记录的控制和状态
    struct Step {
        float actual_steer;   ///< 实际转向角 (rad)
        float actual_speed;   ///< 实际速度 (m/s)
        float x, y, yaw;     ///< Python 端记录的车辆状态（用于验证）
        Polygon vehicle_box;  ///< 该步的车辆包围盒
    };
    std::vector<Step> steps;

    std::string status;       ///< episode 结束状态
    int total_steps;

    VehicleParams vehicle_params;
};

/**
 * 从 JSON 文件加载场景数据
 */
inline SceneData loadScene(const std::string& json_path) {
    std::ifstream ifs(json_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("无法打开场景文件: " + json_path);
    }

    nlohmann::json j;
    ifs >> j;

    SceneData scene;

    // 起点
    scene.start_x   = j["start"][0].get<float>();
    scene.start_y   = j["start"][1].get<float>();
    scene.start_yaw = j["start"][2].get<float>();

    // 终点
    scene.dest_x   = j["dest"][0].get<float>();
    scene.dest_y   = j["dest"][1].get<float>();
    scene.dest_yaw = j["dest"][2].get<float>();

    // 解析多边形的辅助 lambda
    auto parse_polygon = [](const nlohmann::json& arr) -> SceneData::Polygon {
        SceneData::Polygon poly;
        for (const auto& pt : arr) {
            poly.emplace_back(pt[0].get<float>(), pt[1].get<float>());
        }
        return poly;
    };

    // 起始包围盒
    scene.start_box = parse_polygon(j["start_box"]);

    // 目标车位
    scene.parking_lot = parse_polygon(j["parking_lot"]);

    // 障碍物
    for (const auto& obs_json : j["obstacles"]) {
        scene.obstacles.push_back(parse_polygon(obs_json));
    }

    // 步骤数据
    scene.total_steps = j["total_steps"].get<int>();
    scene.status = j["status"].get<std::string>();

    for (const auto& s : j["steps"]) {
        SceneData::Step step;
        step.actual_steer = s["actual_steer"].get<float>();
        step.actual_speed = s["actual_speed"].get<float>();
        step.x   = s["state"][0].get<float>();
        step.y   = s["state"][1].get<float>();
        step.yaw = s["state"][2].get<float>();
        if (s.contains("vehicle_box")) {
            step.vehicle_box = parse_polygon(s["vehicle_box"]);
        }
        scene.steps.push_back(step);
    }

    // 车辆参数
    if (j.contains("vehicle_params")) {
        auto& vp = j["vehicle_params"];
        scene.vehicle_params.wheelbase   = vp.value("wheelbase", 2.8f);
        scene.vehicle_params.front_hang  = vp.value("front_hang", 0.96f);
        scene.vehicle_params.rear_hang   = vp.value("rear_hang", 0.93f);
        scene.vehicle_params.length      = vp.value("length", 4.69f);
        scene.vehicle_params.width       = vp.value("width", 1.94f);
        scene.vehicle_params.step_length = vp.value("step_length", 0.05f);
        scene.vehicle_params.num_step    = vp.value("num_step", 10);
        scene.vehicle_params.mini_iter   = vp.value("mini_iter", 20);
    }

    return scene;
}
