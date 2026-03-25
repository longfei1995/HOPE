#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "bicycle_model.h"
#include "scene_loader.h"

/**
 * 泊车场景可视化器
 *
 * 以车辆后轴中心为原点绘制场景和轨迹曲线。
 * 使用 OpenCV 绘制 2D 图形，输出 PNG 文件。
 */
class Visualizer {
public:
    /**
     * 构造函数
     * @param img_size   输出图像的宽高（正方形）
     * @param scale      每米对应的像素数
     */
    Visualizer(int img_size = 1200, float scale = 40.0f);

    /**
     * 绘制场景俯视图
     *
     * 以指定的后轴中心坐标为原点，绘制:
     * - 障碍物（灰色填充）
     * - 目标车位（绿色边框 + 朝向箭头）
     * - 起始位置（蓝色边框 + 朝向箭头）
     * - 轨迹折线（前进=蓝，后退=红）
     * - 车辆包围盒（半透明，沿轨迹）
     * - 坐标轴 + 网格 + 比例尺
     *
     * @param scene       场景数据
     * @param trajectory  轨迹点序列
     * @param origin_x    原点 x（后轴中心世界坐标，默认=起点）
     * @param origin_y    原点 y
     * @param output_path 输出 PNG 路径
     */
    void drawScene(const SceneData& scene,
                   const std::vector<TrajectoryPoint>& trajectory,
                   float origin_x, float origin_y,
                   const std::string& output_path) const;

    /**
     * 绘制轨迹数值曲线（yaw, kappa, v 随步数变化）
     *
     * @param trajectory  轨迹点序列
     * @param output_path 输出 PNG 路径
     */
    void drawTrajectoryPlots(const std::vector<TrajectoryPoint>& trajectory,
                             const std::string& output_path) const;

private:
    int   img_size_;
    float scale_;   // 像素/米

    /// 世界坐标 → 图像像素坐标（以 origin 为原点）
    cv::Point worldToPixel(float wx, float wy,
                           float origin_x, float origin_y) const;

    /// 绘制带朝向箭头的车辆矩形
    void drawVehicleBox(cv::Mat& img,
                        float x, float y, float yaw,
                        const VehicleParams& vp,
                        const cv::Scalar& color,
                        float origin_x, float origin_y,
                        int thickness = 2,
                        double alpha = 1.0) const;

    /// 绘制多边形
    void drawPolygon(cv::Mat& img,
                     const SceneData::Polygon& poly,
                     const cv::Scalar& color,
                     float origin_x, float origin_y,
                     bool fill = true,
                     int thickness = 2) const;

    /// 绘制坐标网格
    void drawGrid(cv::Mat& img, float origin_x, float origin_y) const;

    /// 绘制朝向箭头
    void drawHeadingArrow(cv::Mat& img,
                          float x, float y, float yaw,
                          float arrow_len,
                          const cv::Scalar& color,
                          float origin_x, float origin_y) const;

    /// 在子图区域绘制一条数值曲线
    void drawCurve(cv::Mat& img,
                   const cv::Rect& region,
                   const std::vector<float>& values,
                   const std::string& title,
                   const std::string& ylabel,
                   const cv::Scalar& color) const;
};
