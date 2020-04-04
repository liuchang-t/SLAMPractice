#ifndef FRAME_H
#define FRAME_H

#include "myslam/common_include.h"
#include "myslam/camera.h"
namespace myslam
{
    class Frame
    {
    public: // 数据成员
        typedef std::shared_ptr<Frame> Ptr;
        unsigned long id_;  // 本帧的ID
        double time_stamp_; // 本帧的时间戳
        SE3d T_c_w_;     // 世界坐标系到相机坐标系的变换矩阵
        Camera::Ptr camera_;    // RGB-D相机模型
        Mat color_, depth_;     // 以Mat类型存储的本帧的彩色 color_ 和深度 depth_ 图像数据

    public: //  方法成员
        Frame();
        Frame(long id, double time_stamp = 0, SE3d T_c_w = SE3d(), Camera::Ptr camera = nullptr, Mat color = Mat(), Mat depth = Mat());
        ~Frame();

        // 创建frame
        static Frame::Ptr createFrame();
        // 获取深度图中的深度信息
        double findDepth(const cv::KeyPoint& kp);
        // 获取相机在世界坐标系中的坐标，也就是相机的位置，就是T_c_w的逆矩阵中的平移向量
        Vector3d getCamCenter() const;
        // 检查一个点是否在这个Frame中，输入参数是该点在世界坐标系下的坐标
        bool isInFrame(const Vector3d& p_w);
    };
}

#endif