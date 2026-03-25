#include "visualizer.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ════════════════════════════════════════════════════════════════════
// 构造函数
// ════════════════════════════════════════════════════════════════════
Visualizer::Visualizer(int img_size, float scale)
    : img_size_(img_size), scale_(scale) {}

// ════════════════════════════════════════════════════════════════════
// 坐标变换：世界坐标 → 图像像素
// ════════════════════════════════════════════════════════════════════
cv::Point Visualizer::worldToPixel(float wx, float wy,
                                   float origin_x, float origin_y) const {
    int px = static_cast<int>((wx - origin_x) * scale_) + img_size_ / 2;
    int py = img_size_ / 2 - static_cast<int>((wy - origin_y) * scale_);
    return {px, py};
}

// ════════════════════════════════════════════════════════════════════
// 绘制多边形
// ════════════════════════════════════════════════════════════════════
void Visualizer::drawPolygon(cv::Mat& img,
                             const SceneData::Polygon& poly,
                             const cv::Scalar& color,
                             float origin_x, float origin_y,
                             bool fill, int thickness) const {
    if (poly.size() < 2) return;

    std::vector<cv::Point> pts;
    pts.reserve(poly.size());
    for (const auto& [wx, wy] : poly) {
        pts.push_back(worldToPixel(wx, wy, origin_x, origin_y));
    }

    if (fill) {
        cv::fillPoly(img, std::vector<std::vector<cv::Point>>{pts}, color);
    } else {
        cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, true, color, thickness);
    }
}

// ════════════════════════════════════════════════════════════════════
// 绘制朝向箭头
// ════════════════════════════════════════════════════════════════════
void Visualizer::drawHeadingArrow(cv::Mat& img,
                                  float x, float y, float yaw,
                                  float arrow_len,
                                  const cv::Scalar& color,
                                  float origin_x, float origin_y) const {
    cv::Point start = worldToPixel(x, y, origin_x, origin_y);
    float ex = x + arrow_len * std::cos(yaw);
    float ey = y + arrow_len * std::sin(yaw);
    cv::Point end = worldToPixel(ex, ey, origin_x, origin_y);
    cv::arrowedLine(img, start, end, color, 2, cv::LINE_AA, 0, 0.3);
}

// ════════════════════════════════════════════════════════════════════
// 绘制车辆包围盒（带旋转的矩形）
// ════════════════════════════════════════════════════════════════════
void Visualizer::drawVehicleBox(cv::Mat& img,
                                float x, float y, float yaw,
                                const VehicleParams& vp,
                                const cv::Scalar& color,
                                float origin_x, float origin_y,
                                int thickness, double alpha) const {
    // 以后轴中心为参考点，构建车辆四个角点
    // 后轴中心位于车辆纵向的 rear_hang 处
    float cos_yaw = std::cos(yaw);
    float sin_yaw = std::sin(yaw);
    float half_w  = vp.width / 2.0f;

    // 车辆四角（局部坐标：x 向前，y 向左）
    // rb=右后, rf=右前, lf=左前, lb=左后
    float corners_local[4][2] = {
        {-vp.rear_hang, -half_w},                    // 右后
        {vp.front_hang + vp.wheelbase, -half_w},     // 右前
        {vp.front_hang + vp.wheelbase,  half_w},     // 左前
        {-vp.rear_hang,  half_w},                    // 左后
    };

    std::vector<cv::Point> pts(4);
    for (int i = 0; i < 4; ++i) {
        float wx = x + corners_local[i][0] * cos_yaw - corners_local[i][1] * sin_yaw;
        float wy = y + corners_local[i][0] * sin_yaw + corners_local[i][1] * cos_yaw;
        pts[i] = worldToPixel(wx, wy, origin_x, origin_y);
    }

    if (alpha < 1.0 - 1e-6) {
        // 半透明绘制
        cv::Mat overlay;
        img.copyTo(overlay);
        cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{pts}, color);
        cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img);
    } else {
        if (thickness > 0) {
            cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, true, color, thickness, cv::LINE_AA);
        } else {
            cv::fillPoly(img, std::vector<std::vector<cv::Point>>{pts}, color);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// 绘制坐标网格
// ════════════════════════════════════════════════════════════════════
void Visualizer::drawGrid(cv::Mat& img, float origin_x, float origin_y) const {
    cv::Scalar grid_color(220, 220, 220);
    cv::Scalar axis_color(150, 150, 150);

    // 计算可见范围（世界坐标）
    float half_range = img_size_ / (2.0f * scale_);
    float x_min = origin_x - half_range;
    float x_max = origin_x + half_range;
    float y_min = origin_y - half_range;
    float y_max = origin_y + half_range;

    // 1m 网格线
    for (float gx = std::floor(x_min); gx <= std::ceil(x_max); gx += 1.0f) {
        cv::Point p1 = worldToPixel(gx, y_min, origin_x, origin_y);
        cv::Point p2 = worldToPixel(gx, y_max, origin_x, origin_y);
        cv::Scalar c = (std::abs(gx - origin_x) < 0.01f) ? axis_color : grid_color;
        int t = (std::abs(gx - origin_x) < 0.01f) ? 2 : 1;
        cv::line(img, p1, p2, c, t);
    }
    for (float gy = std::floor(y_min); gy <= std::ceil(y_max); gy += 1.0f) {
        cv::Point p1 = worldToPixel(x_min, gy, origin_x, origin_y);
        cv::Point p2 = worldToPixel(x_max, gy, origin_x, origin_y);
        cv::Scalar c = (std::abs(gy - origin_y) < 0.01f) ? axis_color : grid_color;
        int t = (std::abs(gy - origin_y) < 0.01f) ? 2 : 1;
        cv::line(img, p1, p2, c, t);
    }

    // 坐标轴标签
    cv::Point center = worldToPixel(origin_x, origin_y, origin_x, origin_y);
    cv::putText(img, "O", {center.x + 5, center.y + 20},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, axis_color, 1);
    cv::putText(img, "x", worldToPixel(origin_x + half_range * 0.9f, origin_y, origin_x, origin_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 1);
    cv::putText(img, "y", worldToPixel(origin_x, origin_y + half_range * 0.9f, origin_x, origin_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 1);

    // 比例尺（右下角）
    int bar_len = static_cast<int>(2.0f * scale_); // 2m
    cv::Point bar_start = {img_size_ - 30 - bar_len, img_size_ - 30};
    cv::Point bar_end   = {img_size_ - 30, img_size_ - 30};
    cv::line(img, bar_start, bar_end, {0, 0, 0}, 2);
    cv::putText(img, "2m", {bar_start.x, bar_start.y - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1);
}

// ════════════════════════════════════════════════════════════════════
// 绘制场景俯视图
// ════════════════════════════════════════════════════════════════════
void Visualizer::drawScene(const SceneData& scene,
                           const std::vector<TrajectoryPoint>& trajectory,
                           float origin_x, float origin_y,
                           const std::string& output_path) const {
    // 白色背景
    cv::Mat img(img_size_, img_size_, CV_8UC3, cv::Scalar(255, 255, 255));

    // 1) 网格
    drawGrid(img, origin_x, origin_y);

    // 2) 障碍物（灰色填充）
    cv::Scalar obstacle_color(180, 180, 180);  // BGR
    for (const auto& obs : scene.obstacles) {
        drawPolygon(img, obs, obstacle_color, origin_x, origin_y, true);
    }

    // 3) 目标车位（绿色边框）
    cv::Scalar dest_color(0, 180, 0);
    drawPolygon(img, scene.parking_lot, dest_color, origin_x, origin_y, false, 3);
    drawHeadingArrow(img, scene.dest_x, scene.dest_y, scene.dest_yaw,
                     1.5f, dest_color, origin_x, origin_y);

    // 4) 起始位置（浅蓝色边框）
    cv::Scalar start_color(237, 149, 100);  // BGR for (100, 149, 237) = cornflower blue
    drawPolygon(img, scene.start_box, start_color, origin_x, origin_y, false, 2);
    drawHeadingArrow(img, scene.start_x, scene.start_y, scene.start_yaw,
                     1.5f, start_color, origin_x, origin_y);

    // 5) 轨迹车辆包围盒（半透明，每隔几步画一次）
    if (!trajectory.empty()) {
        int step_interval = std::max(1, static_cast<int>(trajectory.size()) / 15);
        for (size_t i = 0; i < trajectory.size(); i += step_interval) {
            const auto& pt = trajectory[i];
            // 颜色渐变：从浅到深
            float ratio = static_cast<float>(i) / static_cast<float>(trajectory.size());
            int b = static_cast<int>(255 * (1.0f - ratio));
            int r = static_cast<int>(255 * ratio);
            cv::Scalar traj_color(b, 100, r);
            drawVehicleBox(img, pt.x, pt.y, pt.yaw, scene.vehicle_params,
                           traj_color, origin_x, origin_y, -1, 0.3);
        }
        // 最后一个位置用实线画
        const auto& last = trajectory.back();
        drawVehicleBox(img, last.x, last.y, last.yaw, scene.vehicle_params,
                       {255, 30, 30}, origin_x, origin_y, 2, 1.0);
    }

    // 6) 轨迹折线（颜色编码速度方向）
    for (size_t i = 1; i < trajectory.size(); ++i) {
        cv::Point p1 = worldToPixel(trajectory[i - 1].x, trajectory[i - 1].y,
                                     origin_x, origin_y);
        cv::Point p2 = worldToPixel(trajectory[i].x, trajectory[i].y,
                                     origin_x, origin_y);
        // 前进=蓝色，后退=红色
        cv::Scalar line_color = (trajectory[i].v >= 0)
                                ? cv::Scalar(200, 100, 0)    // 蓝色系
                                : cv::Scalar(0, 50, 220);     // 红色系
        int line_thick = std::max(1, static_cast<int>(std::abs(trajectory[i].v) * 1.2f));
        line_thick = std::min(line_thick, 4);
        cv::line(img, p1, p2, line_color, line_thick, cv::LINE_AA);
    }

    // 7) 标注信息
    std::ostringstream info;
    info << "Status: " << scene.status << "  Steps: " << scene.total_steps;
    cv::putText(img, info.str(), {15, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 2);

    // 图例
    int legend_y = 60;
    cv::line(img, {15, legend_y}, {45, legend_y}, {200, 100, 0}, 2);
    cv::putText(img, "Forward", {50, legend_y + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1);
    legend_y += 25;
    cv::line(img, {15, legend_y}, {45, legend_y}, {0, 50, 220}, 2);
    cv::putText(img, "Backward", {50, legend_y + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1);
    legend_y += 25;
    cv::rectangle(img, {15, legend_y - 8}, {45, legend_y + 8}, {0, 180, 0}, 2);
    cv::putText(img, "Target", {50, legend_y + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1);
    legend_y += 25;
    cv::rectangle(img, {15, legend_y - 8}, {45, legend_y + 8}, {237, 149, 100}, 2);
    cv::putText(img, "Start", {50, legend_y + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1);
    legend_y += 25;
    cv::rectangle(img, {15, legend_y - 8}, {45, legend_y + 8}, {180, 180, 180}, -1);
    cv::putText(img, "Obstacle", {50, legend_y + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1);

    // 原点标记
    cv::Point o = worldToPixel(origin_x, origin_y, origin_x, origin_y);
    cv::drawMarker(img, o, {0, 0, 200}, cv::MARKER_CROSS, 15, 2);

    cv::imwrite(output_path, img);
}

// ════════════════════════════════════════════════════════════════════
// 绘制数值曲线的辅助函数
// ════════════════════════════════════════════════════════════════════
void Visualizer::drawCurve(cv::Mat& img,
                           const cv::Rect& region,
                           const std::vector<float>& values,
                           const std::string& title,
                           const std::string& ylabel,
                           const cv::Scalar& color) const {
    if (values.empty()) return;

    // 边距
    int margin_left   = 70;
    int margin_right  = 20;
    int margin_top    = 35;
    int margin_bottom = 35;

    int plot_w = region.width - margin_left - margin_right;
    int plot_h = region.height - margin_top - margin_bottom;

    float v_min = *std::min_element(values.begin(), values.end());
    float v_max = *std::max_element(values.begin(), values.end());
    if (std::abs(v_max - v_min) < 1e-6f) {
        v_min -= 1.0f;
        v_max += 1.0f;
    }
    // 留 10% margin
    float v_range = v_max - v_min;
    v_min -= v_range * 0.1f;
    v_max += v_range * 0.1f;

    // 绘制背景和边框
    cv::rectangle(img,
                  {region.x + margin_left, region.y + margin_top},
                  {region.x + margin_left + plot_w, region.y + margin_top + plot_h},
                  {240, 240, 240}, -1);
    cv::rectangle(img,
                  {region.x + margin_left, region.y + margin_top},
                  {region.x + margin_left + plot_w, region.y + margin_top + plot_h},
                  {150, 150, 150}, 1);

    // 零线
    if (v_min < 0 && v_max > 0) {
        int zero_y = region.y + margin_top + plot_h -
                     static_cast<int>((0 - v_min) / (v_max - v_min) * plot_h);
        cv::line(img,
                 {region.x + margin_left, zero_y},
                 {region.x + margin_left + plot_w, zero_y},
                 {200, 200, 200}, 1, cv::LINE_AA);
    }

    // 绘制曲线
    for (size_t i = 1; i < values.size(); ++i) {
        float x1_f = static_cast<float>(i - 1) / static_cast<float>(values.size() - 1) * plot_w;
        float x2_f = static_cast<float>(i) / static_cast<float>(values.size() - 1) * plot_w;
        float y1_f = plot_h - (values[i - 1] - v_min) / (v_max - v_min) * plot_h;
        float y2_f = plot_h - (values[i] - v_min) / (v_max - v_min) * plot_h;

        cv::Point p1(region.x + margin_left + static_cast<int>(x1_f),
                     region.y + margin_top + static_cast<int>(y1_f));
        cv::Point p2(region.x + margin_left + static_cast<int>(x2_f),
                     region.y + margin_top + static_cast<int>(y2_f));
        cv::line(img, p1, p2, color, 2, cv::LINE_AA);
    }

    // 标题
    cv::putText(img, title,
                {region.x + margin_left, region.y + margin_top - 10},
                cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 1);

    // Y 轴刻度（5 个）
    for (int i = 0; i <= 4; ++i) {
        float v = v_min + (v_max - v_min) * i / 4.0f;
        int py = region.y + margin_top + plot_h -
                 static_cast<int>(static_cast<float>(i) / 4.0f * plot_h);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << v;
        cv::putText(img, ss.str(), {region.x + 5, py + 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, {80, 80, 80}, 1);
        cv::line(img, {region.x + margin_left - 3, py},
                 {region.x + margin_left, py}, {150, 150, 150}, 1);
    }

    // X 轴标签
    cv::putText(img, "step",
                {region.x + margin_left + plot_w / 2 - 15,
                 region.y + margin_top + plot_h + 28},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {80, 80, 80}, 1);

    // Y 轴标签
    cv::putText(img, ylabel,
                {region.x + 5, region.y + region.height / 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, {80, 80, 80}, 1);
}

// ════════════════════════════════════════════════════════════════════
// 绘制轨迹数值曲线
// ════════════════════════════════════════════════════════════════════
void Visualizer::drawTrajectoryPlots(const std::vector<TrajectoryPoint>& trajectory,
                                     const std::string& output_path) const {
    if (trajectory.empty()) return;

    int plot_w = 900;
    int plot_h = 250;
    int total_h = plot_h * 3 + 20;  // 3 个子图 + 间距
    cv::Mat img(total_h, plot_w, CV_8UC3, cv::Scalar(255, 255, 255));

    // 提取数据
    std::vector<float> yaws, kappas, vs;
    yaws.reserve(trajectory.size());
    kappas.reserve(trajectory.size());
    vs.reserve(trajectory.size());
    for (const auto& pt : trajectory) {
        yaws.push_back(pt.yaw);
        kappas.push_back(pt.kappa);
        vs.push_back(pt.v);
    }

    // 3 个子图
    drawCurve(img, {0, 0, plot_w, plot_h},
              yaws, "Heading (yaw)", "rad", {180, 0, 0});

    drawCurve(img, {0, plot_h, plot_w, plot_h},
              kappas, "Curvature (kappa)", "1/m", {0, 140, 0});

    drawCurve(img, {0, plot_h * 2, plot_w, plot_h},
              vs, "Speed (v)", "m/s", {0, 0, 200});

    cv::imwrite(output_path, img);
}
