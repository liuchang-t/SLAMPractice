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
        // û������������Ǿ������õĲ���
        myslam::Config::setParameterFile("C:/Users/LiuChang/source/repos/SLAMPractice/config/default.yaml");
    }
    else if (argc == 2)
    {
        // �������Ϊ�����ļ�����·����������
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

    // config�ļ��б�����н�Ҫʹ�õ����ݼ��Ĵ洢·������������ǰ�ǵü��config�ļ������Ƿ���ȷ
    string dataset_dir = myslam::Config::get<string>("dataset_dir");
    cout << "dataset: " << dataset_dir << endl;
    // associate.txt�ļ����ǲ�ɫͼ������ͼ���ʱ���������Ϣ������ļ���associate.py�ű���
    // ͨ����ȡ���ݼ��е�rgb.txt��depth.txt�����ļ�֮�󣬶��������ݽ���ƥ��Ȼ������
    ifstream fin(dataset_dir + "/associate.txt");
    if (!fin)
    {
        cout << "please generate the associate file called associate.txt!" << endl;
        return 1;
    }
    // ��associate.txt�ļ���һ��һ�еض�����ɫ�����ͼ���ʱ������ļ����ƣ��ֱ𱣴��ڲ�ͬ��������
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
    // ʹ��vizģ�鴴�����ӻ����ڣ�����ʼ����������ϵ���������ϵ����ʼ�ӽǵȵ�
    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos(0, -1.0, -1.0), cam_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);
    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);

    // ׼�����δ���ͼƬ
    cout << "read total " << rgb_files.size() << " entries" << endl;
    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);
    for (int i = 0; i < rgb_files.size(); i++)
    {
        cout << "****** loop " << i << " ******" << endl;
        // imread ��ȡͼƬ
        Mat color = cv::imread(rgb_files[i]);
        Mat depth = cv::imread(depth_files[i], -1);
        if (color.data == nullptr || depth.data == nullptr)
            break;
        // ����ȡ����ͼ�����ݺ�ʱ�������Ϊһ�� Frame ֡����
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        // boost::timer timer_1;
        // �� VisualOdometry �����м���� Frame ֡
        // addFrame()��VisualOdometry����Ҫ�������������������ȡ�������Ӽ��㡢ͼ��ƥ�䡢λ�˹��ơ��ο�֡���ݸ��µ�һϵ������
        vo->addFrame(pFrame);
        // cout<<"VO costs time: "<<timer_1.elapsed()<<endl;

        // ��һ�μ��飬���vo���ڸ�����״̬�����˳�ѭ��
        if (vo->state_ == myslam::VisualOdometry::LOST)
            break;
        // ���vo��û�и������Ǿͻ�ȡ��ǰ�������������ϵ�е�λ�ˣ���addFrame()��ʱ�򣬻���µ�ǰ֡��Tcw����Tcw���������
        // ����������ϵ���������ϵ��λ�˱任����������ת�����ƽ��������ɣ�������ת�����������������ϵ���������ϵ�µ���̬��
        // ��ƽ���������������������ϵ��ԭ�����������ϵ�е����ꡣ
        // ��ˣ���Tcw������󣬾Ϳ��Եõ��������ϵ����������ϵ�µ�λ�ˡ�
        SE3d Twc = pFrame->T_c_w_.inverse();

        // show the map and the camera pose 
        // ��Ҫʹ��cv::Affine3��������vizģ������ϵλ�õĸ��£�cv::Affine3����ĺ�����ʵ����һ��4*4��λ�˱任���󣬸����䶨�壬
        // ����ʹ����ת����R��ƽ������t��Ϊ�����������������ֻ��������Ҫ�ľ�����������£�
        // ����SE3d
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

        // ��������զ��ɫͼ���ϱ�ǳ���
        Mat img_show = color.clone();
        for (auto& pt : vo->map_->map_points_)
        {
            myslam::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel(p->pos_, pFrame->T_c_w_);
            cv::circle(img_show, cv::Point2f(pixel(0, 0), pixel(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("image", img_show);   // ��ʾ�����������ǵ�ͼ��
        cv::waitKey(1); 
        vis.setWidgetPose("Camera", M);
        vis.spinOnce(1, false);
    }

    return 0;
}
