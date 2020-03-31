﻿#include "myslam/frame.h"

namespace myslam
{
    Frame::Frame()
        : id_(-1), time_stamp_(-1), camera_(nullptr)
    {}

    Frame::Frame(long id, double time_stamp, SE3d T_c_w, Camera::Ptr camera, Mat color, Mat depth)
        : id_(id), time_stamp_(time_stamp), T_c_w_(T_c_w), camera_(camera), color_(color), depth_(depth)
    {}

    Frame::~Frame()
    {}

    Frame::Ptr Frame::createFrame()
    {
        static long factory_id = 0;
        return Frame::Ptr(new Frame(factory_id++));
    }

    double Frame::findDepth(const cv::KeyPoint& kp)
    {
        int x = cvRound(kp.pt.x);
        int y = cvRound(kp.pt.y);
        ushort d = depth_.ptr<ushort>(y)[x];
        if (d != 0)
        {
            return double(d) / camera_->depth_scale_;
        }
        else
        {
            // check the nearby points 
            int dx[4] = { -1,0,1,0 };
            int dy[4] = { 0,-1,0,1 };
            for (int i = 0; i < 4; i++)
            {
                d = depth_.ptr<ushort>(y + dy[i])[x + dx[i]];
                if (d != 0)
                {
                    return double(d) / camera_->depth_scale_;
                }
            }
        }
        return -1.0;
    }


    Vector3d Frame::getCamCenter() const
    {
        return T_c_w_.inverse().translation();
    }

    bool Frame::isInFrame(const Vector3d& pt_world)
    {
        Vector3d p_cam = camera_->world2camera(pt_world, T_c_w_);
        // 如果该点在相机坐标系下的z坐标（即深度）是负的，那肯定不在这一帧内
        if (p_cam(2, 0) < 0)
            return false;
        // 如果z坐标是正的，那计算它在像素坐标系中的坐标，如果x、y坐标均为正值，
        // 且均小于彩色图像的列数和行数，就代表它在这一帧图像的取像范围内
        Vector2d pixel = camera_->world2pixel(pt_world, T_c_w_);
        return pixel(0, 0) > 0 && pixel(1, 0) > 0
            && pixel(0, 0) < color_.cols
            && pixel(1, 0) < color_.rows;
    }
}
