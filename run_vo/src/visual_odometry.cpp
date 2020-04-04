#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <algorithm>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

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
        map_point_erase_ratio_ = Config::get<double>("map_point_erase_ratio");
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
            // extract features from first frame 
            extractKeyPoints();
            computeDescriptors();
            addKeyFrame();      // the first frame is a key-frame
            break;
        }
        case OK:
        {
            curr_ = frame;
            curr_->T_c_w_ = ref_->T_c_w_;
            extractKeyPoints();
            computeDescriptors();
            featureMatching();
            poseEstimationPnP();
            bool check = checkEstimatedPose();
            if (check) // a good estimation
            {
                // 更新当前帧的Tcw矩阵，
                curr_->T_c_w_ = T_c_w_estimated_;
                optimizeMap();
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

    // 根据当前帧和地图中的特征点描述子信息，进行特征匹配
    // 输出：match_3dpts_, match_2dkp_index_
    void VisualOdometry::featureMatching()
    {
        /* 
        1. desp_map是用于此次匹配的地图点描述子集合，由于地图点会越来越多，而实际上能够与当前帧进行匹配的地图点只是其中的一部分，
        因此可以进行筛选，仅把有可能与当前帧构成匹配关系的地图点取出来，减少匹配算法的工作量。
        2. 具体方法就是通过检查地图点的坐标是否可能在当前帧的视野范围内，包括深度是否有效（>=0），以及映射到像素坐标系之后，是否在成像范围内。
        3. 由于此时当前帧还未进行位姿估计，因此没有自己的T_c_w_，实际上是使用的参考帧的位姿矩阵来进行地图点的 world2pixel 坐标变换，
        这就是为什么在此之前有一步是 curr_->T_c_w_ = ref_->T_c_w_;
        */
        Mat desp_map;
        vector<MapPoint::Ptr> candidate;
        for (auto& allpoints : map_->map_points_)
        {
            MapPoint::Ptr& p = allpoints.second;
            // check if p in curr frame image 
            if (curr_->isInFrame(p->pos_))
            {
                // add to candidate 
                p->visible_times_++;
                candidate.push_back(p);
                desp_map.push_back(p->descriptor_);
            }
        }

        // 下面这一段是使用surf特征进行匹配的测试
        //int minHessian = 450;
        //cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
        //vector<KeyPoint> keypoints_src;
        //vector<KeyPoint> keypoints_dst;
        //Mat descriptor_src, descriptor_dst;
        //detector->detectAndCompute(curr_->color_, Mat(), keypoints_src, descriptor_src);
        //detector->detectAndCompute(ref_->color_, Mat(), keypoints_dst, descriptor_dst);
        //vector<cv::DMatch> matches_surf;
        //cv::FlannBasedMatcher matcher_flann__surf;
        //matcher_flann__surf.match(descriptor_src, descriptor_dst, matches_surf);


        vector<cv::DMatch> matches;
        //matcher_flann_.match(desp_map, descriptors_curr_, matches);   // ORB特征不能使用默认构造的FlannBasedMatcher

        //基于FLANN的描述符对象匹配，当使用ORB特征的时候，需要重新构造HASH
        cv::Ptr<DescriptorMatcher> matcher = makePtr<FlannBasedMatcher>(makePtr<flann::LshIndexParams>(12, 20, 2));
        //匹配
        matcher->match(desp_map, descriptors_curr_, matches);

        // select the best matches
        float min_dis = std::min_element(
            matches.begin(), matches.end(),
            [](const cv::DMatch& m1, const cv::DMatch& m2)
            {
                return m1.distance < m2.distance;
            })->distance;

        match_3dpts_.clear();
        match_2dkp_index_.clear();
        for (cv::DMatch& m : matches)
        {
            if (m.distance < max<float>(min_dis * match_ratio_, 30.0))
            {
                match_3dpts_.push_back(candidate[m.queryIdx]);
                match_2dkp_index_.push_back(m.trainIdx);
            }
        }
        cout << "good matches: " << match_3dpts_.size() << endl;
    }

    // 根据特征匹配的结果
    // 当前帧关键点坐标keypoints_curr_，以及相机内参，估计当前帧在世界坐标系下的位姿T_c_w_estimated_
    void VisualOdometry::poseEstimationPnP()
    {
        // construct the 3d 2d observations
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;
        // 根据特征匹配结果，统计所有的匹配到的特征点在当前帧中的图像坐标
        for (int index : match_2dkp_index_)
        {
            pts2d.push_back(keypoints_curr_[index].pt);
        }
        // 根据特征匹配结果，统计所有的匹配到的特征点在地图中的坐标
        for (MapPoint::Ptr pt : match_3dpts_)
        {
            pts3d.push_back(pt->getPositionCV());
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

        // 为rvec和tvec设定初始值，看看能不能增加成功率
        Eigen::AngleAxisd r_v = Eigen::AngleAxisd(ref_->T_c_w_.rotationMatrix());
        Eigen::Vector3d r_vector3d(r_v.angle()*r_v.axis());
        for (int i = 0; i < 3; i++)
        {
            rvec.at<double>(i) = r_vector3d(i);
            tvec.at<double>(i) = ref_->T_c_w_.translation()(i);
        }

        cv::solvePnPRansac(pts3d, pts2d, K, distCoeffs, rvec, tvec, true, 100, 4.0, 0.99, inliers);
        num_inliers_ = inliers.rows;
        // 打印输出PnP求解的内点数量
        cout<<"pnp inliers: "<<num_inliers_<<endl;

        // 将Vector3d类型表示的旋转向量转换为 Eigen::Quaterniond ，然后才能用来构造 SO3 
        // 步骤：先把 Vector3d 类型的旋转向量的模和方向提取出来构造 AngleAxis 类型的旋转向量，
        // 然后再从 AngleAxis 转到 Quaterniond 
        Eigen::Vector3d v_r(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0));
        Eigen::Vector3d v_t(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0));
        Eigen::AngleAxisd rotation_vector(v_r.norm(), v_r / v_r.norm());
        Eigen::Quaterniond quater = Eigen::Quaterniond(rotation_vector);
        ////////////////////////////////////////////////
        // 其实有了四元数或者旋转矩阵之后，就可以和平移向量一起直接构造SE3
        //cout << "v_t = " << v_t.transpose() << ",  v_r = " << v_r.transpose() << endl;
        //cout << "v_r.norm = " << v_r.norm() << endl;
        T_c_w_estimated_ = SE3d(quater, v_t);

        //现在已经通过PnP估计出了T_c_w，但是为了提高精度，我们进一步使用g2o对相机位姿进行优化
        //注意这里仅优化位姿，而不优化地图点
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
        Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        Block* solver_ptr = new Block(std::unique_ptr<Block::LinearSolverType>(linearSolver));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<Block>(solver_ptr));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(
            T_c_w_estimated_.rotationMatrix(),
            T_c_w_estimated_.translation()
            ));
        optimizer.addVertex(pose);

        // edges
        for (int i = 0; i < inliers.rows; i++)
        {
            int index = inliers.at<int>(i, 0);
            // 3D -> 2D projection
            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0, pose);
            edge->camera_ = curr_->camera_.get();
            edge->point_ = Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
            edge->setMeasurement(Vector2d(pts2d[index].x, pts2d[index].y));
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
            // set the inlier map points 
            match_3dpts_[index]->matched_times_++;
        }

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        T_c_w_estimated_ = SE3d(
            pose->estimate().rotation(),
            pose->estimate().translation()
            );
        cout << "T_c_w_estimated_: " << endl << T_c_w_estimated_.matrix() << endl;
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
        // 这里应该判断的是当前帧相对上一帧是否运动过大，如果过大，就要放弃，因此要看的是Trc的值
        //Sophus::Vector6d d = T_c_w_estimated_.log();
        Sophus::Vector6d d = (ref_->T_c_w_ * T_c_w_estimated_.inverse()).log();
        //Vector3d trans = d.head<3>();
        if (d.norm() > 1)
        {
            cout << "reject because motion is too large: " << d.norm() << endl;
            return false;
        }
        return true;
    }

    // 检查当前帧是否是关键帧，判断策略是检查相对参考帧（上一个关键帧）的运动是否足够大，
    // 只有运动足够大才算关键帧
    bool VisualOdometry::checkKeyFrame()
    {
        // 计算T_r_c = T_r_w*T_w_c
        SE3d T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
        // 通过.log()函数把位姿变换矩阵 T_r_c 转化为旋量表示
        Sophus::Vector6d d = T_r_c.log();
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
        if (map_->keyframes_.empty())
        {
            // 当前帧是第一帧，把所有有效特征点（深度大于0）都添加到地图中
            // first key-frame, add all 3d points into map
            for (size_t i = 0; i < keypoints_curr_.size(); i++)
            {
                // 判断该特征点的深度值是否有效
                double d = curr_->findDepth(keypoints_curr_[i]);
                if (d < 0)
                    continue;
                Vector3d p_world = curr_->camera_->pixel2world(
                    Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), curr_->T_c_w_, d
                    );
                // 获取观测方向向量
                Vector3d n = p_world - curr_->getCamCenter();
                n.normalize();
                // 将该特征点创建为地图点
                MapPoint::Ptr map_point = MapPoint::createMapPoint(
                    p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
                    );
                // 将地图点添加到地图中
                map_->insertMapPoint(map_point);
            }
        }
        // 将该帧作为关键帧，插入到地图中
        map_->insertKeyFrame(curr_);
        // 将最新的关键帧设定为以后的参考帧
        ref_ = curr_;
    }

    void VisualOdometry::addMapPoints()
    {
        // add the new map points into map
        // 将当前帧特征点中那些没有匹配到地图点的特征点找出来
        vector<bool> matched(keypoints_curr_.size(), false);
        for (int index : match_2dkp_index_)
            matched[index] = true;
        for (int i = 0; i < keypoints_curr_.size(); i++)
        {
            // 将当前帧特征点中那些没有匹配到地图点的特征点找出来
            if (matched[i] == true)
                continue;
            // 如果他们是有效特征点（深度 >= 0，这里是用参考帧深度判断的，修改为当前帧）
            //double d = ref_->findDepth(keypoints_curr_[i]);
            double d = curr_->findDepth(keypoints_curr_[i]);
            if (d < 0)
                continue;
            // 计算他们的世界坐标（这里使用了当前帧的Tcw，但用的是参考帧的d，修改为当前帧）
            Vector3d p_world = curr_->camera_->pixel2world(
                Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y),
                curr_->T_c_w_, d
                );
            // 计算他们的视线方向向量（这里使用了参考帧的相机中心坐标，修改为当前帧）
            //Vector3d n = p_world - ref_->getCamCenter();
            Vector3d n = p_world - curr_->getCamCenter();
            n.normalize();
            // 创建mappoint，并插入到地图中
            MapPoint::Ptr map_point = MapPoint::createMapPoint(
                p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
                );
            map_->insertMapPoint(map_point);
        }
        cout << "map points: " << map_->map_points_.size() << endl;
    }

    void VisualOdometry::optimizeMap()
    {
        // remove the hardly seen and no visible points 
        for (auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
        {
            // 剔除不在当前帧视野范围内的地图点
            if (!curr_->isInFrame(iter->second->pos_))
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            // 剔除那些虽然在视野中，但是总是无法成功与图像匹配的地图点
            float match_ratio = float(iter->second->matched_times_) / iter->second->visible_times_;
            if (match_ratio < map_point_erase_ratio_)
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            // 剔除那些参考帧和当前帧观察角度相差过大的地图点，mappoint类的norm_成员（视线方向）都是相对于当初创建它的那一帧相机位置计算的
            double angle = getViewAngle(curr_, iter->second);
            if (angle > M_PI / 6.)
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
            if (iter->second->good_ == false)
            {
                // TODO try triangulate this map point 
            }
            iter++;
        }
        // 如果当前帧匹配到的地图点数量小于100，就执行一遍添加地图点的程序
        if (match_2dkp_index_.size() < 100)
            addMapPoints();
        if (map_->map_points_.size() > 1000)
        {
            // TODO map is too large, remove some one 
            map_point_erase_ratio_ += 0.05;
        }
        else
            map_point_erase_ratio_ = 0.1;
        cout << "map points: " << map_->map_points_.size() << endl;
    }

    double VisualOdometry::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point)
    {
        Vector3d n = point->pos_ - frame->getCamCenter();
        n.normalize();
        return acos(n.transpose() * point->norm_);
    }
}