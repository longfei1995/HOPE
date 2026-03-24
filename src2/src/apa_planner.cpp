#include "apa_planner.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

// ─── 构造函数 ─────────────────────────────────────────────────────
APAPlanner::APAPlanner(const std::string& model_path, bool deterministic)
    : deterministic_(deterministic) {
    try {
        model_ = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        throw std::runtime_error("无法加载模型: " + std::string(e.what()));
    }
    model_.eval();
    initDiscreteActions();
}

// ─── 初始化离散动作表 ──────────────────────────────────────────────
void APAPlanner::initDiscreteActions() {
    // 与 configs.py 的离散动作生成逻辑一致:
    //   PRECISION=10, step_speed=1, VALID_STEER=0.75
    //   steer: 0.75, 0.675, 0.6, ..., 0, ..., -0.675, -0.75 (共21个)
    //   前 21 个 speed=+1（前进），后 21 个 speed=-1（后退），共 42 个
    //
    // choose_action 中的归一化: steer / STEER_SCALE, speed / 1
    constexpr int   PRECISION  = 10;
    constexpr float STEER_MAX  = 0.75f;
    constexpr float STEER_STEP = STEER_MAX / PRECISION;  // 0.075
    constexpr float SPEED_FWD  = 1.0f;
    constexpr float SPEED_BWD  = -1.0f;

    discrete_actions_.clear();
    discrete_actions_.reserve(N_ACTIONS);

    // 前进动作 (21个)
    for (int i = 0; i <= 2 * PRECISION; ++i) {
        float raw_steer = STEER_MAX - i * STEER_STEP;
        float scaled_steer = raw_steer / STEER_SCALE;  // 归一化到 [-1, 1]
        discrete_actions_.push_back({scaled_steer, SPEED_FWD});
    }
    // 后退动作 (21个)
    for (int i = 0; i <= 2 * PRECISION; ++i) {
        float raw_steer = STEER_MAX - i * STEER_STEP;
        float scaled_steer = raw_steer / STEER_SCALE;
        discrete_actions_.push_back({scaled_steer, SPEED_BWD});
    }
}

// ─── 动作选择 ──────────────────────────────────────────────────────
int APAPlanner::selectAction(float mean0, float mean1,
                             float std0, float std1,
                             const std::vector<float>& action_mask) {
    // 与 Python ActionMask.choose_action 一致的高斯概率计算
    constexpr float LOG_2PI = 1.8378770664093453f;
    std::vector<float> log_probs(N_ACTIONS);

    for (int i = 0; i < N_ACTIONS; ++i) {
        float z0 = (discrete_actions_[i][0] - mean0) / std0;
        float z1 = (discrete_actions_[i][1] - mean1) / std1;
        float lp0 = -0.5f * z0 * z0 - 0.5f * LOG_2PI - std::log(std0);
        float lp1 = -0.5f * z1 * z1 - 0.5f * LOG_2PI - std::log(std1);
        lp0 = std::clamp(lp0, -10.0f, 10.0f);
        lp1 = std::clamp(lp1, -10.0f, 10.0f);
        log_probs[i] = lp0 + lp1;
    }

    // exp(log_prob) * action_mask
    std::vector<float> weighted(N_ACTIONS);
    for (int i = 0; i < N_ACTIONS; ++i) {
        weighted[i] = std::exp(log_probs[i]) * action_mask[i];
    }

    if (deterministic_) {
        // 确定性模式：选概率最大的合法动作
        return static_cast<int>(
            std::distance(weighted.begin(),
                          std::max_element(weighted.begin(), weighted.end())));
    } else {
        // 随机采样模式
        float sum = std::accumulate(weighted.begin(), weighted.end(), 0.0f);
        if (sum < 1e-10f) {
            // 所有动作都被掩蔽，均匀随机
            static std::mt19937 rng(42);
            return std::uniform_int_distribution<int>(0, N_ACTIONS - 1)(rng);
        }
        static std::mt19937 rng(42);
        std::discrete_distribution<int> dist(weighted.begin(), weighted.end());
        return dist(rng);
    }
}

// ─── 主推理入口 ────────────────────────────────────────────────────
APAPlanner::PlanResult APAPlanner::run(
    const std::vector<float>& lidar,
    const std::vector<float>& target,
    const std::vector<float>& img,
    const std::vector<float>& action_mask)
{
    // 输入维度校验
    if (static_cast<int>(lidar.size()) != LIDAR_DIM)
        throw std::invalid_argument("lidar 维度应为 120，实际: " + std::to_string(lidar.size()));
    if (static_cast<int>(target.size()) != TARGET_DIM)
        throw std::invalid_argument("target 维度应为 5，实际: " + std::to_string(target.size()));
    if (static_cast<int>(img.size()) != IMG_C * IMG_H * IMG_W)
        throw std::invalid_argument("img 维度应为 12288，实际: " + std::to_string(img.size()));
    if (static_cast<int>(action_mask.size()) != N_ACTIONS)
        throw std::invalid_argument("action_mask 维度应为 42，实际: " + std::to_string(action_mask.size()));

    // 构造输入张量（batch_size = 1）
    auto options = torch::TensorOptions().dtype(torch::kFloat32);

    torch::Tensor lidar_t = torch::from_blob(
        const_cast<float*>(lidar.data()), {1, LIDAR_DIM}, options).clone();
    torch::Tensor target_t = torch::from_blob(
        const_cast<float*>(target.data()), {1, TARGET_DIM}, options).clone();
    torch::Tensor img_t = torch::from_blob(
        const_cast<float*>(img.data()), {1, IMG_C, IMG_H, IMG_W}, options).clone();
    torch::Tensor mask_t = torch::from_blob(
        const_cast<float*>(action_mask.data()), {1, N_ACTIONS}, options).clone();

    // 前向推理
    torch::NoGradGuard no_grad;
    auto output = model_.forward({lidar_t, target_t, img_t, mask_t});
    auto tuple_out = output.toTuple();
    torch::Tensor mean_t = tuple_out->elements()[0].toTensor();   // (1, 2)
    torch::Tensor std_t  = tuple_out->elements()[1].toTensor();   // (1, 2)

    float mean0 = mean_t[0][0].item<float>();
    float mean1 = mean_t[0][1].item<float>();
    float std0  = std_t[0][0].item<float>();
    float std1  = std_t[0][1].item<float>();

    // 动作掩码后处理：从 42 个离散动作中选择
    int idx = selectAction(mean0, mean1, std0, std1, action_mask);

    PlanResult result;
    result.action_idx   = idx;
    result.steer        = discrete_actions_[idx][0];    // [-1, 1]
    result.speed        = discrete_actions_[idx][1];    // {-1, 1}
    result.actual_steer = result.steer * STEER_SCALE;   // rad
    result.actual_speed = result.speed * SPEED_SCALE;    // m/s
    return result;
}
