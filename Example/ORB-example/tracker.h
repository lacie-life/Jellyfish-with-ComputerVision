//
// Created by lacie on 18/04/2021.
//

#ifndef ORB_EXAMPLE_TRACKER_H
#define ORB_EXAMPLE_TRACKER_H

#include <opencv2/features2d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>      //for imshow

#include <vector>
#include <iostream>
#include <iomanip>

#include "utils.h" // Drawing and printing function

using namespace std;
using namespace cv;

const double akaze_thresh = 3e-4; // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio
const int bb_min_inliers = 100; // Minimal number of inliers to draw bounding box
const int stats_update_period = 10; // On-screen statistics are updated every 10 frames

class Tracker{
    public:
        Tracker(Ptr<Feature2D> _detector, Ptr<DescriptorMatcher> _matcher):
                detector(_detector),
                matcher(_matcher)
        {}

        void setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats);
        Mat process(const Mat frame, Stats& stats);
        Ptr<Feature2D> getDtector(){
            return detector;
        }

    protected:
        Ptr<Feature2D> detector;
        Ptr<DescriptorMatcher> matcher;
        Mat first_frame, first_desc;
        vector<KeyPoint> first_kp;
        vector<Point2f> object_bb;
    };
void Tracker::setFirstFrame(const Mat frame, vector<Point2f> bb, string title, Stats& stats)
{
    cv::Point *ptMask = new cv::Point[bb.size()];
    const Point* ptContain = { &ptMask[0] };
    int iSize = static_cast<int>(bb.size());
    for (size_t i=0; i<bb.size(); i++) {
        ptMask[i].x = static_cast<int>(bb[i].x);
        ptMask[i].y = static_cast<int>(bb[i].y);
    }
    first_frame = frame.clone();
    cv::Mat matMask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::fillPoly(matMask, &ptContain, &iSize, 1, cv::Scalar::all(255));
    detector->detectAndCompute(first_frame, matMask, first_kp, first_desc);
    stats.keypoints = (int)first_kp.size();
    drawBoundingBox(first_frame, bb);
    putText(first_frame, title, Point(0, 60), FONT_HERSHEY_PLAIN, 5, Scalar::all(0), 4);
    object_bb = bb;
    delete[] ptMask;
}

Mat Tracker::process(const Mat frame, Stats &stats) {
    TickMeter tm;
    vector<KeyPoint> kp;
    Mat desc;

    tm.start();
    detector->detectAndCompute(frame, noArray(), kp, desc);

    stats.keypoints = (int)kp.size();

    vector<vector<DMatch>> matches;
    vector<KeyPoint> matched_1, matched_2;

    matcher->knnMatch(first_desc, desc, matches, 2);

    for(unsigned  i = 0; i < matches.size(); i++){
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance){
            matched_1.push_back(first_kp[matches[i][0].queryIdx]);
            matched_2.push_back(kp[matches[i][0].trainIdx]);
        }
    }
    stats.matches = (int)matched_1.size();

    Mat inlier_mask, homography;
    vector<KeyPoint> inliers_1, inliers_2;
    vector<DMatch> inlier_matches;

    if(matched_1.size() >= 4){
        homography = findHomography(Points(matched_1),
                                    Points(matched_2),
                                    RANSAC, ransac_thresh,
                                    inlier_mask);
    }
    tm.stop();
    stats.fps = 1. / tm.getTimeSec();

    if(matched_1.size() < 4 || homography.empty()){
        Mat res;
        hconcat(first_frame, frame, res);
        stats.inliers = 0;
        stats.ratio = 0;
        return res;
    }

    for (unsigned i = 0; i < matched_1.size(); i++){
        if(inlier_mask.at<uchar>(i)){
            int new_i = static_cast<int>(inliers_1.size());
            inliers_1.push_back(matched_1[i]);
            inliers_2.push_back(matched_2[i]);
            inlier_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }
    stats.inliers = (int)inliers_1.size();
    stats.ratio = stats.inliers * 1.0 / stats.matches;

    vector<Point2f> new_bb;
    perspectiveTransform(object_bb, new_bb, homography);
    Mat frame_with_bb = frame.clone();

    if(stats.inliers >= bb_min_inliers){
        drawBoundingBox(frame_with_bb, new_bb);
    }
    Mat res;
    drawMatches(first_frame, inliers_1,
                frame_with_bb, inliers_2,
                inlier_matches, res,
                Scalar(255, 0, 0),
                Scalar (255, 0, 0));
    return res;
}
#endif //ORB_EXAMPLE_TRACKER_H
