// -------------- test the visual odometry -------------
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
//#include <boost/timer/timer.hpp>
#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main(int argc, char** argv)
{
    cout << "argc = " << argc << endl;
    if (argc == 1)
    {
        // 没有输入参数，那就用内置的参数
        myslam::Config::setParameterFile("C:/Users/LiuChang/source/repos/SLAMPractice/config/default.yaml");
    }
    else if (argc == 2)
    {
        // 输入参数为参数文件完整路径包括名称
        myslam::Config::setParameterFile(argv[1]);
    }
    else
    {
        cout << "usage: run_vo parameter_file" << endl;
        return 1;
    }

    myslam::Camera::Ptr camera(new myslam::Camera(
        myslam::Config::get<float>("camera.fx"),
        myslam::Config::get<float>("camera.fy"),
        myslam::Config::get<float>("camera.cx"),
        myslam::Config::get<float>("camera.cy"),
        myslam::Config::get<float>("camera.depth_scale")));

    // config文件中保存的有将要使用的数据集的存储路径，程序运行前记得检查config文件内容是否正确
    string dataset_dir = myslam::Config::get<string>("dataset_dir");
    cout << "dataset: " << dataset_dir << endl;
    // associate.txt文件中是彩色图像和深度图像的时间戳对齐信息，这个文件由associate.py脚本，
    // 通过读取数据集中的rgb.txt和depth.txt两个文件之后，对其中内容进行匹配然后生成
    ifstream fin(dataset_dir + "/associate.txt");
    if (!fin)
    {
        cout << "please generate the associate file called associate.txt!" << endl;
        return 1;
    }
    // 从associate.txt文件中一行一行地读出彩色和深度图像的时间戳的文件名称，分别保存在不同的数组里
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while (!fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        depth_files.push_back(dataset_dir + "/" + depth_file);

        if (fin.good() == false)
            break;
    }

    // visualization
    // 使用viz模块创建可视化窗口，并初始化世界坐标系、相机坐标系、初始视角等等
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos(0, -1.0, -1.0), cam_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);

    // 准备依次处理图片
    cout << "read total " << rgb_files.size() << " entries" << endl;
    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);
    for (int i = 0; i < rgb_files.size(); i++)
    {
        cout << "****** loop " << i << " ******" << endl;
        // imread 读取图片
        Mat color = cv::imread(rgb_files[i]);
        Mat depth = cv::imread(depth_files[i], -1);
        if (color.data == nullptr || depth.data == nullptr)
            break;
        // 将读取到的图像数据和时间戳整合为一个 Frame 帧对象
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        // boost::timer timer_1;
        // 向 VisualOdometry 对象中加入该 Frame 帧
        // addFrame()是VisualOdometry的主要函数，其包含了特征提取、描述子计算、图像匹配、位姿估计、参考帧数据更新等一系列流程
        vo->addFrame(pFrame);
        // cout<<"VO costs time: "<<timer_1.elapsed()<<endl;

        // 做一次检验，如果vo处于跟丢的状态，就退出循环
        if (vo->state_ == myslam::VisualOdometry::LOST)
            break;
        // 如果vo还没有跟丢，那就获取当前相机在世界坐标系中的位姿，在addFrame()的时候，会更新当前帧的Tcw矩阵，Tcw矩阵代表了
        // 从世界坐标系到相机坐标系的位姿变换矩阵，他由旋转矩阵和平移向量组成，其中旋转矩阵代表了世界坐标系在相机坐标系下的姿态，
        // 而平移向量则代表了世界坐标系的原点在相机坐标系中的坐标。
        // 因此，对Tcw求逆矩阵，就可以得到相机坐标系在世界坐标系下的位姿。
        SE3d Twc = pFrame->T_c_w_.inverse();

        // show the map and the camera pose 
        // 需要使用cv::Affine3类来进行viz模块坐标系位置的更新，cv::Affine3对象的核心其实就是一个4*4的位姿变换矩阵，根据其定义，
        // 可以使用旋转矩阵R和平移向量t作为参数来声明这个对象，只不过它需要的矩阵和向量如下：
        // 由于SE3d
        cv::Affine3d M(
            cv::Affine3d::Mat3(
                Twc.rotationMatrix()(0, 0), Twc.rotationMatrix()(0, 1), Twc.rotationMatrix()(0, 2),
                Twc.rotationMatrix()(1, 0), Twc.rotationMatrix()(1, 1), Twc.rotationMatrix()(1, 2),
                Twc.rotationMatrix()(2, 0), Twc.rotationMatrix()(2, 1), Twc.rotationMatrix()(2, 2)
                ),
            cv::Affine3d::Vec3(
                Twc.translation()(0, 0), Twc.translation()(1, 0), Twc.translation()(2, 0)
                )
            );

        // 将特征点咋彩色图像上标记出来
        Mat img_show = color.clone();
        for (auto& pt : vo->map_->map_points_)
        {
            myslam::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel(p->pos_, pFrame->T_c_w_);
            cv::circle(img_show, cv::Point2f(pixel(0, 0), pixel(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("image", img_show);   // 显示带有特征点标记的图像
        cv::waitKey(1); 
        vis.setWidgetPose("Camera", M);
        vis.spinOnce(1, false);
    }

    return 0;
}
