#pragma once

#include <cmath>
#include <vector>

/**
 * 自行车运动学模型（Kinematic Single-Track Model）
 *
 * 与 Python 端 KSModel 完全对齐：
 *   x  += v * cos(θ) * dt_micro
 *   y  += v * sin(θ) * dt_micro
 *   θ  += v * tan(δ) / L * dt_micro
 *
 * 其中 dt_micro = step_length / mini_iter
 */

/// 轨迹点：包含位姿、曲率和速度
struct TrajectoryPoint {
    float x;        ///< 后轴中心 x (m)
    float y;        ///< 后轴中心 y (m)
    float yaw;      ///< 航向角 (rad)
    float kappa;    ///< 曲率 (1/m)，= tan(steer) / wheelbase
    float v;        ///< 速度 (m/s)
};

/// 车辆参数（与 configs.py 对齐）
struct VehicleParams {
    float wheelbase   = 2.8f;
    float front_hang  = 0.96f;
    float rear_hang   = 0.93f;
    float length      = 4.69f;
    float width       = 1.94f;
    float step_length = 0.05f;   // 每微步时间 (s)
    int   num_step    = 10;      // 每个决策步的微步数
    int   mini_iter   = 20;      // 每个微步内的数值积分次数
};

/// 控制输入
struct ControlInput {
    float steer;    ///< 前轮转向角 (rad)，范围 [-0.75, 0.75]
    float speed;    ///< 速度 (m/s)，范围 [-2.5, 2.5]
};

/**
 * 自行车运动学模型
 */
class BicycleModel {
public:
    explicit BicycleModel(const VehicleParams& params = VehicleParams())
        : params_(params) {}

    /**
     * 执行一个决策步的状态更新（等价于 Python KSModel.step）
     *
     * @param x0, y0, yaw0  初始状态
     * @param ctrl           控制输入 (steer, speed)
     * @return               更新后的轨迹点
     */
    TrajectoryPoint step(float x0, float y0, float yaw0,
                         const ControlInput& ctrl) const {
        float x   = x0;
        float y   = y0;
        float yaw = yaw0;

        // 限幅
        float steer = std::clamp(ctrl.steer, -0.75f, 0.75f);
        float speed = std::clamp(ctrl.speed, -2.5f, 2.5f);

        float dt_micro = params_.step_length / static_cast<float>(params_.mini_iter);

        for (int s = 0; s < params_.num_step; ++s) {
            for (int m = 0; m < params_.mini_iter; ++m) {
                x   += speed * std::cos(yaw) * dt_micro;
                y   += speed * std::sin(yaw) * dt_micro;
                yaw += speed * std::tan(steer) / params_.wheelbase * dt_micro;
            }
        }

        float kappa = std::tan(steer) / params_.wheelbase;

        return {x, y, yaw, kappa, speed};
    }

    /**
     * 从初始状态出发，依次执行一系列控制输入，生成完整轨迹
     *
     * @param x0, y0, yaw0  初始状态
     * @param controls       控制输入序列
     * @return               轨迹点序列（包含初始点，长度 = controls.size() + 1）
     */
    std::vector<TrajectoryPoint> propagate(
        float x0, float y0, float yaw0,
        const std::vector<ControlInput>& controls) const
    {
        std::vector<TrajectoryPoint> traj;
        traj.reserve(controls.size() + 1);

        // 初始点（无控制输入，kappa 和 v 设为 0）
        traj.push_back({x0, y0, yaw0, 0.0f, 0.0f});

        float x = x0, y = y0, yaw = yaw0;
        for (const auto& ctrl : controls) {
            auto pt = step(x, y, yaw, ctrl);
            traj.push_back(pt);
            x   = pt.x;
            y   = pt.y;
            yaw = pt.yaw;
        }
        return traj;
    }

    const VehicleParams& params() const { return params_; }

private:
    VehicleParams params_;
};
