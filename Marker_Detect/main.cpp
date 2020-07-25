#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>

void createAcuro(){
    cv::Mat markerImage;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::drawMarker(dictionary, 24, 200, markerImage, 1);
    cv::imwrite("marker24.png", markerImage);
}

int dictitonaryId = 7;
float marker_length_m = 0.1;
int wait_time = 10;

cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << 614.5685, 0.0, 323.7408, 0.0, 614.4438, 243.1708, 0.0, 0.0, 1.0);

cv::Mat dist_coeffs = (cv::Mat_<double>(12,1) << 614.5685, 0.0, 323.7408, 0.0, 0.0, 614.4438, 243.1708, 0.0, 0.0, 0.0, 1.0, 0.0);

int main()
{
    createAcuro();

    std::cout << "camera_matrix \n" << camera_matrix << std::endl;
    std::cout << camera_matrix.size() << std::endl;

    std::cout << "\ndist_coeffs \n" << dist_coeffs << std::endl;
    std::cout << dist_coeffs.size() << std::endl;

    rs2::pipeline pipe;

    rs2::config cfg;
 //   cfg.enable_stream(RS2_STREAM_ANY, 640, 480, RS2_FORMAT_BGR8, 30);

 //   pipe.start(cfg);
    pipe.start();

    cv::Mat Image;
    std::cout << "King" << std::endl;

  //  auto const i = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>().get_intrinsics();

    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary( \
        cv::aruco::PREDEFINED_DICTIONARY_NAME(dictitonaryId));

    std::ostringstream vector_to_marker;

    while(1)
    {
        // Block program until frames arrive
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();

        cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::cvtColor(color, Image, CV_BGR2RGB);

        cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display Image", Image);

        cv::Mat image_copy;
        Image.copyTo(image_copy);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(Image, dictionary, corners, ids);
        
        // If at least one marker detected
    //    if (ids.size() > 0)

    //        cv::aruco::drawDetectedMarkers(image_copy, corners, ids);

    //    cv::imshow("Detected markers", image_copy);

        if (ids.size() > 0)
        {
            cv::aruco::drawDetectedMarkers(image_copy, corners, ids);
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_m,
                    camera_matrix, dist_coeffs, rvecs, tvecs);
            
        //    std::cout << "Bug" << std::endl;
            // Draw axis for each marker
            for(int i=0; i < ids.size(); i++)
            {
                cv::aruco::drawAxis(image_copy, camera_matrix, dist_coeffs,
                        rvecs[i], tvecs[i], 0.1);

                // This section is going to print the data for all the detected
                // markers. If you have more than a single marker, it is
                // recommended to change the below section so that either you
                // only print the data for a specific marker, or you print the
                // data for each marker separately.
                vector_to_marker.str(std::string());
                vector_to_marker << std::setprecision(4)
                                 << "x: " << std::setw(8) << tvecs[0](0);
                cv::putText(image_copy, vector_to_marker.str(),
                            cvPoint(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cvScalar(0, 252, 124), 1, CV_AA);

                vector_to_marker.str(std::string());
                vector_to_marker << std::setprecision(4)
                                 << "y: " << std::setw(8) << tvecs[0](1);
                cv::putText(image_copy, vector_to_marker.str(),
                            cvPoint(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cvScalar(0, 252, 124), 1, CV_AA);

                vector_to_marker.str(std::string());
                vector_to_marker << std::setprecision(4)
                                 << "z: " << std::setw(8) << tvecs[0](2);
                cv::putText(image_copy, vector_to_marker.str(),
                            cvPoint(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cvScalar(0, 252, 124), 1, CV_AA);
            }
        }

        imshow("Pose estimation", image_copy);

        char c=(char)cv::waitKey(25);
        if(c==27)
            break;
    }

    return EXIT_SUCCESS;

}


