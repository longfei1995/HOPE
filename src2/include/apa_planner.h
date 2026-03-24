#pragma once

#include <torch/script.h>
#include <array>
#include <string>
#include <vector>

/**
 * APA (Automated Parking Assist) 规划器
 *
 * 加载 TorchScript 导出的 SAC actor 网络，
 * 接收多模态观测，输出泊车控制动作。
 */
class APAPlanner {
public:
    /// 规划结果
    struct PlanResult {
        float steer;           ///< 归一化转向角 [-1, 1]
        float speed;           ///< 归一化速度 {-1, 1}
        float actual_steer;    ///< 实际转向角 (rad)，范围 [-0.75, 0.75]
        float actual_speed;    ///< 实际速度 (m/s)，范围 [-2.5, 2.5]
        int   action_idx;      ///< 选择的离散动作索引 [0, 41]
    };

    /**
     * 构造函数
     * @param model_path  TorchScript 模型文件路径（由 export_torchscript.py 生成）
     * @param deterministic  true = 选概率最大的动作；false = 按概率采样
     */
    explicit APAPlanner(const std::string& model_path, bool deterministic = true);

    /**
     * 执行一次规划
     * @param lidar       激光雷达数据，长度 120
     * @param target      目标位姿 [x, y, theta, cos_theta, sin_theta]，长度 5
     * @param img         摄像头图像，通道优先 (3×64×64)，长度 12288
     * @param action_mask 动作掩码，长度 42，每个值 ∈ [0, 1]
     * @return PlanResult 规划结果
     */
    PlanResult run(const std::vector<float>& lidar,
                   const std::vector<float>& target,
                   const std::vector<float>& img,
                   const std::vector<float>& action_mask);

private:
    torch::jit::script::Module model_;
    bool deterministic_;

    /// 42 个离散动作 (归一化后: steer ∈ [-1,1], speed ∈ {-1, 1})
    std::vector<std::array<float, 2>> discrete_actions_;

    static constexpr float STEER_SCALE = 0.75f;   ///< steer 缩放系数
    static constexpr float SPEED_SCALE = 2.5f;     ///< speed 缩放系数
    static constexpr int   N_ACTIONS   = 42;
    static constexpr int   LIDAR_DIM   = 120;
    static constexpr int   TARGET_DIM  = 5;
    static constexpr int   IMG_C       = 3;
    static constexpr int   IMG_H       = 64;
    static constexpr int   IMG_W       = 64;

    /// 初始化 42 个离散动作表
    void initDiscreteActions();

    /**
     * 基于策略分布和动作掩码选择离散动作
     * @return 选中的动作索引
     */
    int selectAction(float mean0, float mean1,
                     float std0, float std1,
                     const std::vector<float>& action_mask);
};
