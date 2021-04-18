#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>      //for imshow

#include <vector>
#include <iostream>

#include "tracker.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    string video_name = "../data/blais.mp4";

    VideoCapture video_in;
    video_in.open(video_name);

    Stats stats, akaze_stats, orb_stats;

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->setThreshold(akaze_thresh);

    Ptr<ORB> orb = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    Tracker akaze_tracker(akaze, matcher);
    Tracker orb_tracker(orb, matcher);

    Mat frame;
    namedWindow(video_name, WINDOW_NORMAL);

    cout<< "\n Press any key to stop the video and select a bounding box" << endl;

    while(waitKey(1) < 1){
        video_in >> frame;
        cv::resizeWindow(video_name, frame.size());
        imshow(video_name, frame);
    }

    vector<Point2f> bb;

    cv::Rect uBox = cv::selectROI(video_name, frame);
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x+uBox.width), static_cast<float>(uBox.y)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x + uBox.width), static_cast<float>(uBox.y + uBox.height)));
    bb.push_back(cv::Point2f(static_cast<float>(uBox.x), static_cast<float>(uBox.y, uBox.height)));

    akaze_tracker.setFirstFrame(frame, bb, "AKAZE", stats);
    orb_tracker.setFirstFrame(frame, bb, "ORB", stats);

    Stats akaze_draw_stats, orb_draw_stats;
    Mat akaze_res, orb_res, res_frame;

    int i = 0;

    for(;;){
        i++;
        bool update_stats = (i % stats_update_period == 0);
        video_in  >> frame;
        // stop the program if no more image
        if(frame.empty()) break;

        akaze_res = akaze_tracker.process(frame, stats);
        akaze_stats += stats;

        if(update_stats){
            akaze_draw_stats = stats;
        }
        orb->setMaxFeatures(stats.keypoints);
        orb_res = orb_tracker.process(frame, stats);
        orb_stats += stats;

        if(update_stats){
            orb_draw_stats = stats;
        }

        drawStatistics(akaze_res, akaze_draw_stats);
        drawStatistics(orb_res, orb_draw_stats);
        vconcat(akaze_res, orb_res, res_frame);
        cv::imshow(video_name, res_frame);
        if(waitKey(1) == 27){
            break;
        } // quit on ESC button
    }
    akaze_stats /= i - 1;
    orb_stats /= i - 1;
    printStatistics("AKAZE", akaze_stats);
    printStatistics("ORB", orb_stats);

    return 0;
}