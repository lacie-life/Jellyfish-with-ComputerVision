#include <iostream>
#include <algorithm>
#include <fstream>

#include <librealsense2/rs.hpp>

#include <opencv2/opencv.hpp>
#include <thread>

#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

#include <Eigen/Eigen>

using namespace cv;
using namespace std;

#define MAX_DISTANCE 1.5

float detR(float H[16]) {
    return H[0] * (H[5] * H[10] - H[9] * H[6]) - H[4] * (H[1] * H[10] - H[2] * H[9]) + H[8] * (H[1] * H[6] - H[5] * H[2]);
}

void quat2rot(rs2_quaternion &q, Eigen::Matrix4f &m){
    m = Eigen::Matrix4f::Identity();
    // (row, col)
    m(0, 0) = 1 - 2 * q.y*q.y - 2 * q.z*q.z;
    m(1, 0) = 2 * q.x*q.y + 2 * q.z*q.w;
    m(2, 0) = 2 * q.x*q.z - 2 * q.y*q.w;

    m(0, 1) = 2 * q.x*q.y - 2 * q.z*q.w;
    m(1, 1) = 1 - 2 * q.x*q.x - 2 * q.z*q.z;
    m(2, 1) = 2 * q.y*q.z + 2 * q.x*q.w;

    m(0, 2) = 2 * q.x*q.z + 2 * q.y*q.w;
    m(1, 2) = 2 * q.y*q.z - 2 * q.x*q.w;
    m(2, 2) = 1 - 2 * q.x*q.x - 2 * q.y*q.y;
}

void get_t265_d435i(float H_t265_d435i[16], Eigen::Matrix4f &m){
    m = Eigen::Matrix4f::Identity();
    // (row, column)
    m(0, 0) = H_t265_d435i[0];
    m(1, 0) = H_t265_d435i[1];
    m(2, 0) = H_t265_d435i[2];
    m(3, 0) = H_t265_d435i[3];

    m(0, 1) = H_t265_d435i[4];
    m(1, 1) = H_t265_d435i[5];
    m(2, 1) = H_t265_d435i[6];
    m(3, 1) = H_t265_d435i[7];

    m(0, 2) = H_t265_d435i[8];
    m(1, 2) = H_t265_d435i[9];
    m(2, 2) = H_t265_d435i[10];
    m(3, 2) = H_t265_d435i[11];

    m(0, 3) = H_t265_d435i[12];
    m(1, 3) = H_t265_d435i[13];
    m(2, 3) = H_t265_d435i[14];
    m(3, 3) = H_t265_d435i[15];
}

void draw_pointcloud(pcl::visualization::PCLVisualizer::Ptr &viewer,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr,
                     rs2::points& points, rs2::video_frame& color,
                     rs2_pose& pose,
                     float H_t265_d435i[16]){

    auto vertices = points.get_vertices();
    auto tex_coords = points.get_texture_coordinates();

    // T265 pose
    Eigen::Matrix4f transform_world_t265 = Eigen::Matrix4f::Identity();
    quat2rot(pose.rotation, transform_world_t265);
    transform_world_t265(0, 3) = pose.translation.x;
    transform_world_t265(1, 3) = pose.translation.y;
    transform_world_t265(2, 3) = pose.translation.z;

    // T265 to D435i entrinsics
    Eigen::Matrix4f transform_t265_d435i = Eigen::Matrix4f::Identity();
    get_t265_d435i(H_t265_d435i, transform_t265_d435i);

    // Check
    //std::cout << "World T265: \n" << transform_world_t265 << std::endl;
    //std::cout << "T265 -> D435i: \n" << transform_t265_d435i << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cur_frame_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < points.size(); i++){
        if(vertices[i].z > 0 && vertices[i].z < MAX_DISTANCE){
            cur_frame_cloud->points.push_back(pcl::PointXYZ(vertices[i].x,
                                                            vertices[i].y,
                                                            vertices[i].z));
        }
    }

    pcl::transformPointCloud(*cur_frame_cloud, *cur_frame_cloud, transform_t265_d435i);
    pcl::transformPointCloud(*cur_frame_cloud, *cur_frame_cloud, transform_world_t265);

    *point_cloud_ptr += *cur_frame_cloud;

    pcl::PointCloud<pcl::PointXYZ>::Ptr output(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(point_cloud_ptr);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*output);

    *point_cloud_ptr = *output;

    viewer->updatePointCloud(output, "cloud");
    viewer->spinOnce(30);
}


int main(int argc, char** argv) try {
    std::cout << "Hello from RealSense !!!" << std::endl;

    rs2::pointcloud pc;
    rs2::points points;
    rs2::pose_frame pose_frame(nullptr);
    rs2::video_frame color_frame(nullptr);

    rs2::context ctx;
    std::vector<rs2::pipeline> pipelines;

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Object Sacnning"));
    viewer->addPointCloud<pcl::PointXYZ>(point_cloud_ptr, "cloud");

    // Start a streaming pipe per each connected device
    for(auto&& dev : ctx.query_devices()){
        std::cout << dev.get_info(RS2_CAMERA_INFO_NAME) << " "
                  << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << " "
                  << dev.get_info(RS2_CAMERA_INFO_PRODUCT_ID) << std::endl;

        rs2::pipeline pipe(ctx);
        rs2::config cfg;
        cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));

        pipe.start(cfg);
        pipelines.emplace_back(pipe);
    }

    // entrinsics
    // depth w.r.t. tracking (column-major)
    float H_t265_d435i[16] = {1, 0, 0, 0,
                              0, -1, 0, 0,
                              0, 0, -1, 0,
                              0, 0, 0, 1};

    std::string fn = "../H_T265_D435i.cfg";
    std::ifstream ifs(fn);
    if (!ifs.is_open()) {
        std::cerr << "Couldn't open " << fn << std::endl;
        return -1;
    }
    else {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                ifs >> H_t265_d435i[i + 4 * j];  // row-major to column-major
            }
        }
    }

    float det = detR(H_t265_d435i);
    if(fabs(1-det) > 1e-6){
        std::cerr << "Invalid homogeneous transformation matrix input (det != 1)" << std::endl;
        return -1;
    }

    int saved = 0;
    cv::namedWindow("object-scanning");

    while (!viewer->wasStopped()){

        // Collect data
        for (auto &&pipe : pipelines) {
            auto frames = pipe.wait_for_frames();
            std::cout << "Hello" << std::endl;

            // color
            auto color = frames.get_color_frame();
            if (!color)
                color = frames.get_infrared_frame();
            if (color) {
                pc.map_to(color);
                color_frame = color;
            }

            // depth
            auto depth_frame = frames.get_depth_frame();
            if (depth_frame)
                points = pc.calculate(depth_frame);

            // pose
            auto pose = frames.get_pose_frame();
            if (pose) {
                pose_frame = pose;

                // Print the x, y, z values of the translation, relative to initial position
                auto pose_data = pose.get_pose_data();
                //std::cout << "\r" << "Device Position: " << std::setprecision(3) << std::fixed << pose_data.translation.x << " " << pose_data.translation.y << " " << pose_data.translation.z << " (meters)" << endl;
            }
        }

        // Visualize
        if(color_frame){
            Mat color(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
            // cvtColor(color, color, CV_RGB2BGR);
            imshow("object-scanning", color);
        }

        int k = cv::waitKey(10);
        if(k == int('q')){
            break;
        }

        if (points && pose_frame && k == int('d')) {
            rs2_pose pose = pose_frame.get_pose_data();
            draw_pointcloud(viewer, point_cloud_ptr, points, color_frame, pose, H_t265_d435i);
        }

        if (k == int('s')) {
            std::string name = "saved_pcd_" + std::to_string(saved++) + ".pcd";
            pcl::io::savePCDFile(name, *point_cloud_ptr);
            std::cerr << "Saved " << point_cloud_ptr->points.size() << " data points to " << name << std::endl;
        }

    }

    return 0;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
