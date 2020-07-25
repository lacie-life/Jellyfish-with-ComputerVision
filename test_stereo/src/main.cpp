#include "../include/StereoCalib.hpp"
#include "../include/StereoVision.hpp"

#include <librealsense2/rs.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>

int count = 0;

void show(std::string title, cv::Mat image1, cv::Mat image2)
{
    int size;
    int i;
    int m, n;
    int x, y;

    // w : Maximum number of image in a row
    // h : Maximum number of image in a column
    int w, h;

    // scale : How much wa have to resize the image

    float scale;
    int max;

    // Two image : size = 300, w = 2, h = 1
    size = 500;
    w = 2;
    h = 1;

    // Create a new 3 channel image
    cv::Mat DispImage = cv::Mat::zeros(cv::Size(100 + size*w, 60 + size*h), CV_8UC3);

    for (int i = 0, m = 20, n = 20; i < 2; i++, m += (20 + size))
    {
        if (i == 0)
        {
            cv::Mat img = image1;

            if(img.empty())
            {
                std::cout << "Invalid arguments" << "\n";
                return;
            }

            // Find the width and height of the image
            x = img.cols;
            y = img.rows;

            // Find whether hieght or width is greater in order to resize the image
            max = (x > y)? x: y;

            // Find the scaling factor to resize image
            scale = (float) ((float)max / size);

            // Used to Align the image
            if (i % w == 0 && m != 20)
            {
                m = 20;
                n += 20 + size;
            }

            // Set the image ROI to display the current image
            // Resize the input image and copy the it to the Single Big Image
            cv::Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
            cv::Mat temp; 
            resize(img,temp, cv::Size(ROI.width, ROI.height));
            temp.copyTo(DispImage(ROI));
        }
        if (i == 1)
        {
            cv::Mat img = image2;

            if(img.empty())
            {
                std::cout << "Invalid arguments" << "\n";
                return;
            }

            // Find the width and height of the image
            x = img.cols;
            y = img.rows;

            // Find whether hieght or width is greater in order to resize the image
            max = (x > y)? x: y;

            // Find the scaling factor to resize image
            scale = (float) ((float)max / size);

            // Used to Align the image
            if (i % w == 0 && m != 20)
            {
                m = 20;
                n += 20 + size;
            }

            // Set the image ROI to display the current image
            // Resize the input image and copy the it to the Single Big Image
            cv::Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
            cv::Mat temp; 
            resize(img,temp, cv::Size(ROI.width, ROI.height));
            temp.copyTo(DispImage(ROI));
        }
    }

    // Create a new window, and show the Single Big Image
    cv::namedWindow( title, 1 );
    cv::imshow( title, DispImage);
    
    char c = (char)cv::waitKey(25);
    if(c == 27)
        exit(0);
    else if (c == 115)
    {
        std::cout << "King" << "\n";

        std::string count_s = std::to_string(count);
        std::string right_path = "test_photos/right-" + count_s + ".jpg";
        std::string left_path = "test_photos/left-" + count_s + ".jpg";
    //    std::string merge_path = "Merge/" + count_s + ".jpg";

        cv::imwrite(right_path, image1);
        cv::imwrite(left_path, image2);
    //    cv::imwrite(merge_path, DispImage);

        StereoVision sv(left_path, right_path);
        sv.getQMatrix();

        std::cout << "X, Y and Z coordinates of a real-world object" << std::endl;
        std::cout << sv.calculateDistance(sv.detectObject()) << std::endl;

        count++;
    }

    // cv::waitKey(0);
}

int main() {

    // Get RealSense Device
    rs2::context ctx;
    auto list = ctx.query_devices();
    if(list.size() == 0)
        throw std::runtime_error("No device decteted");
    rs2::device dev = list.front();

    std::vector<rs2::pipeline> pipelines;

    //Create a configuration for configuring the pipeline with a non default profile
    // rs2::config cfg;

    for (auto&& dev : ctx.query_devices())
    {
        rs2::pipeline pipe(ctx);
        rs2::config cfg;
        cfg.enable_device(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
        pipe.start(cfg);
        pipelines.emplace_back(pipe);
    }

    //Add desired streams to configuration
    // cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    //Instruct pipeline to start streaming with the requested configuration
    // pipe.start(cfg);

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    // rs2::frameset frames;

    // We'll keep track of the last frame of each stream available to make the presentation persistent
    std::map<int, rs2::frame> render_frames;

    // Collect the new frames from all the connected devices
    std::vector<rs2::frame> new_frames;
    cv::Mat Right;
    cv::Mat Left;
    while(1)
    {
        rs2::frameset fs;
        
        if (pipelines.at(0).poll_for_frames(&fs))
        {
            //Get each frame
            rs2::frame color_frame = fs.get_color_frame();
            cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::cvtColor(color, Left, CV_BGR2RGB);

            // Display in a GUI

            // cv::namedWindow("Display Image Left", cv::WINDOW_AUTOSIZE );
            // imshow("Display Image Left", Left);
            // cv::waitKey(0);
        }

        if (pipelines.at(1).poll_for_frames(&fs))
        {
            //Get each frame
            rs2::frame color_frame = fs.get_color_frame();
            cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
            cv::cvtColor(color, Right, CV_BGR2RGB);

            // Display in a GUI

            // cv::namedWindow("Display Image Right", cv::WINDOW_AUTOSIZE );
            // imshow("Display Image Right", Right);
            // cv::waitKey(0);
        }
        
        if (!Right.empty() && !Left.empty())
        {
            show ("Mutil Camera", Right, Left);
        }  
    }
    
//    StereoCalib sc("additional_files/left.txt", "additional_files/right.txt");

//    sc.leftCameraCalibrate();
//    sc.getIntrinsicCoeffsLeft();

//    sc.leftCameraUndistort();
//    sc.getDistortionCoeffsLeft();

//    sc.rightCameraCalibrate();
//    sc.getIntrinsicCoeffsRight();

//    sc.rightCameraUndistort();
//    sc.getDistortionCoeffsRight();
    
//    sc.stereoCalibrateAndRectify();
    
    

//    StereoVision sv("test_photos/left-1.jpg", "test_photos/right-1.jpg");
//    sv.getQMatrix();

//    std::cout << "X, Y and Z coordinates of a real-world object" << std::endl;
//    std::cout << sv.calculateDistance(sv.detectObject()) << std::endl;

    return 0;
}
