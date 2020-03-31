#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

namespace myslam
{
    VisualOdometry::VisualOdometry() :
        state_(INITIALIZING), ref_(nullptr), curr_(nullptr), map_(new Map), num_lost_(0), num_inliers_(0)
    {
        num_of_features_ = Config::get<int>("number_of_features");
        scale_factor_ = Config::get<double>("scale_factor");
        level_pyramid_ = Config::get<int>("level_pyramid");
        match_ratio_ = Config::get<float>("match_ratio");
        max_num_lost_ = Config::get<float>("max_num_lost");
        min_inliers_ = Config::get<int>("min_inliers");
        key_frame_min_rot = Config::get<double>("keyframe_rotation");
        key_frame_min_trans = Config::get<double>("keyframe_translation");
        orb_ = cv::ORB::create(num_of_features_, scale_factor_, level_pyramid_);
    }

    VisualOdometry::~VisualOdometry()
    {

    }

    bool VisualOdometry::addFrame(Frame::Ptr frame)
    {
        switch (state_)
        {
        case INITIALIZING:
        {
            state_ = OK;
            curr_ = ref_ = frame;
            map_->insertKeyFrame(frame);
            // extract features from first frame 
            extractKeyPoints();
            computeDescriptors();
            // compute the 3d position of features in ref frame 
            setRef3DPoints();
            break;
        }
        case OK:
        {
            curr_ = frame;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            if (checkEstimatedPose() == true) // a good estimation
            {
                // 更新当前帧的Tcw矩阵，T_c_w = T_c_r*T_r_w 
                curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
                ref_ = curr_;
                setRef3DPoints();
                num_lost_ = 0;
                if (checkKeyFrame() == true) // is a key-frame
                {
                    addKeyFrame();
                }
            }
            else // bad estimation due to various reasons
            {
                num_lost_++;
                if (num_lost_ > max_num_lost_)
                {
                    state_ = LOST;
                }
                return false;
            }
            break;
        }
        case LOST:
        {
            cout << "vo has lost." << endl;
            break;
        }
        }

        return true;
    }

    // 根据当前帧的 color_ 图像信息，提取出所有的特征点，存储到 keypoints_curr_ 数组中
    void VisualOdometry::extractKeyPoints()
    {
        orb_->detect(cv::InputArray(curr_->color_), keypoints_curr_);
    }

    // 根据提取出的特征点，计算当前帧所有特征点的描述子信息，存储到 descriptors_curr_ 数组中
    void VisualOdometry::computeDescriptors()
    {
        orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
    }

    // 根据当前帧和参考帧的特征点描述子信息，进行特征匹配，将匹配结果存储到 feature_matches_ 数组中
    void VisualOdometry::featureMatching()
    {
        // match desp_ref and desp_curr, use OpenCV's brute force match 
        vector<cv::DMatch> matches;
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        matcher.match(descriptors_ref_, descriptors_curr_, matches);
        // select the best matches
        float min_dis = std::min_element(
            matches.begin(), matches.end(),
            [](const cv::DMatch& m1, const cv::DMatch& m2)
            {
                return m1.distance < m2.distance;
            })->distance;

        feature_matches_.clear();
        for (cv::DMatch& m : matches)
        {
            if (m.distance < max<float>(min_dis * match_ratio_, 30.0))
            {
                feature_matches_.push_back(m);
            }
        }
        // cout<<"good matches: "<<feature_matches_.size()<<endl;
    }

    // setRef3DPoints()函数将当前帧的有效关键点的3D坐标和描述子信息依次存储到 pts_3d_ref_ 和 descriptors_ref_ 数组中
    // 这个3D坐标是在对应帧的相机坐标系下的表示
    void VisualOdometry::setRef3DPoints()
    {
        // select the features with depth measurements 
        pts_3d_ref_.clear();
        descriptors_ref_ = Mat();
        // 对每一个关键点进行操作
        for (size_t i = 0; i < keypoints_curr_.size(); i++)
        {
            // 获取图像上这个点的深度信息
            double d = ref_->findDepth(keypoints_curr_[i]);
            // 只有深度信息大于0的才算有效
            if (d > 0)
            {
                // 获得这个点在相机坐标系下的完整3d坐标
                Vector3d p_cam = ref_->camera_->pixel2camera(
                    Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), d
                    );
                // 把这个特征点的3d坐标加入 pts_3d_ref_ 数组
                pts_3d_ref_.push_back(cv::Point3f(p_cam(0, 0), p_cam(1, 0), p_cam(2, 0)));
                // 把这个点的描述子加入 descriptors_ref_ 数组，描述子对应的是 descriptors_curr_ 的第i行
                descriptors_ref_.push_back(descriptors_curr_.row(i));
            }
        }
    }
    // 根据特征匹配的结果 feature_matches_，参考帧关键点位置pts_3d_ref_，
    // 当前帧关键点坐标keypoints_curr_，以及相机内参，估计参考帧到当前帧的相机位姿变换矩阵 T_c_r_estimated_
    void VisualOdometry::poseEstimationPnP()
    {
        // construct the 3d 2d observations
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;
        // 根据特征匹配的结果索引，统计所有的匹配到的特征点在参考帧相机坐标系中的3D坐标以及在当前帧中的图像坐标
        for (cv::DMatch m : feature_matches_)
        {
            pts3d.push_back(pts_3d_ref_[m.queryIdx]);
            pts2d.push_back(keypoints_curr_[m.trainIdx].pt);
        }
        // 相机内参矩阵
        Mat K = (cv::Mat_<double>(3, 3) <<
            ref_->camera_->fx_, 0, ref_->camera_->cx_,
            0, ref_->camera_->fy_, ref_->camera_->cy_,
            0, 0, 1
            );
        // rvec是旋转向量，tvec是平移向量，inliers是通过RANSAC求解拟合出的有效数据点（内点），即拟合误差小于设定阈值（4.0）的点
        Mat inliers;
        cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);    // output translation vector
        cv::solvePnPRansac(pts3d, pts2d, K, distCoeffs, rvec, tvec, false, 100, 4.0, 0.99, inliers);
        num_inliers_ = inliers.rows;
        // 打印输出PnP求解的内点数量
        // cout<<"pnp inliers: "<<num_inliers_<<endl;

        // 将Vector3d类型表示的旋转向量转换为 Eigen::Quaterniond ，然后才能用来构造 SO3 
        // 步骤：先把 Vector3d 类型的旋转向量的模和方向提取出来构造 AngleAxis 类型的旋转向量，
        // 然后再从 AngleAxis 转到 Quaterniond 
        Eigen::Vector3d v_r(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0));
        Eigen::Vector3d v_t(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));
        Eigen::AngleAxisd rotation_vector(v_r.norm(), v_r / v_r.norm());
        Eigen::Quaterniond quater = Eigen::Quaterniond(rotation_vector);
        ////////////////////////////////////////////////
        // 其实有了四元数或者旋转矩阵之后，就可以和平移向量一起直接构造SE3
        //cout << "v_t = " << v_t.transpose() << "        v_r = " << v_r.transpose() << endl;
        //cout << "v_r.norm = " << v_r.norm() << endl;
        T_c_r_estimated_ = SE3d(quater, v_t);

    }

    // 检查位姿估计是否正确，判断策略是内点数量是否足够，以及运动是否过大
    bool VisualOdometry::checkEstimatedPose()
    {
        // check if the estimated pose is good
        if (num_inliers_ < min_inliers_)
        {
            cout << "reject because inlier is too small: " << num_inliers_ << endl;
            return false;
        }
        // if the motion is too large, it is probably wrong
        Sophus::Vector6d d = T_c_r_estimated_.log();
        if (d.norm() > 5.0)
        {
            cout << "reject because motion is too large: " << d.norm() << endl;
            return false;
        }
        return true;
    }

    // 检查当前帧是否是关键帧，判断策略是检查平移向量和旋转向量的模是否大于阈值，
    // 只有运动足够大才算关键帧
    bool VisualOdometry::checkKeyFrame()
    {
        // 通过.log()函数把位姿变换矩阵转化为旋量表示
        Sophus::Vector6d d = T_c_r_estimated_.log();
        // 旋量的前三个元素是平移向量，后三个元素是旋转向量
        Vector3d trans = d.head<3>();
        Vector3d rot = d.tail<3>();
        if (rot.norm() > key_frame_min_rot || trans.norm() > key_frame_min_trans)
            return true;
        return false;
    }

    // 将当前帧作为关键帧插入到 map_ 成员中
    void VisualOdometry::addKeyFrame()
    {
        cout << "adding a key-frame" << endl;
        map_->insertKeyFrame(curr_);
    }
}