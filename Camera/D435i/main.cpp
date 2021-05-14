#include <librealsense2/rs.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>  

int main(int argc, char * argv[])
{
	int width = 1280;
	int height = 720;
	int fps = 30;
	rs2::config config;
	config.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
	//config.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
    config.enable_stream(RS2_STREAM_COLOR,0 , width, height, RS2_FORMAT_RGB8, fps);

	// start pipeline
	rs2::pipeline pipeline;
	rs2::pipeline_profile pipeline_profile = pipeline.start(config);

    rs2::device selected_device = pipeline_profile.get_device();

    auto depth_sensor = selected_device.first<rs2::depth_sensor>();
    if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED))
    {
        //depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
        depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
    }
    

	while (1) // Application still alive?
	{
		// wait for frames and get frameset
		rs2::frameset frameset = pipeline.wait_for_frames();
        

		// get single infrared frame from frameset
		//rs2::video_frame ir_frame = frameset.get_infrared_frame();

		// get left and right infrared frames from frameset
		rs2::video_frame ir_frame_left = frameset.get_infrared_frame(1);
		//rs2::video_frame ir_frame_right = frameset.get_infrared_frame(2);
        rs2::video_frame color_frame = frameset.get_color_frame();

		cv::Mat dMat_left = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)ir_frame_left.get_data());
		//cv::Mat dMat_right = cv::Mat(cv::Size(width, height), CV_8UC1, (void*)ir_frame_right.get_data());
        cv::Mat color_image = cv::Mat(cv::Size(width, height), CV_8UC3, (void*)color_frame.get_data());

        //cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGB);

        cv::cvtColor(color_image, color_image, cv::COLOR_BGR2RGBA);
        cv::imshow("origin_rgb", color_image);

        std::vector<cv::Mat> channels(4);

        cv::split(color_image, channels);

        channels[3] = dMat_left;

        cv::merge(channels, color_image);
        cv::imshow("color", color_image);
        //std::cout << color_image.type() << std::endl;

        std::vector<cv::Mat> channels_final(4);
        cv::split(color_image, channels_final);
        //cv::imshow("img_r", channels_final[0]);
		//cv::imshow("img_g", channels_final[1]);
        //cv::imshow("img_b", channels_final[2]);
        cv::imshow("img_ir", channels_final[3]);

		char c = cv::waitKey(1);
        if (c == 'q')
			break;
	}

	return EXIT_SUCCESS;
}