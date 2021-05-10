#include <iostream>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>

#include <opencv2/features2d.hpp>
#include <vector>

#define KNN 0

using namespace cv;
using namespace cuda;
using namespace std;

int main(){
    std::cout << "Hello, World!" << std::endl;

    int64 start, end;
    double time;

    cv::Mat img_1 = imread("../images/example-1.jpg");
    cv::Mat img_2 = imread("../images/example-2.jpg");

    if(!img_1.data || ! img_2.data){
        cout << "Error reading images" << endl;
        return -1;
    }

    start = getTickCount();

    vector<Point2f> scense;
    std::vector<KeyPoint> keypoints_1, keypoints_2;

    cuda::GpuMat d_img1, d_img2;
    cuda::GpuMat d_srcL, d_srcR;

    d_img1.upload(img_1);
    d_img2.upload(img_2);

    cv::Mat imgMatches, des_L, des_R;

    cuda::cvtColor(d_img1, d_srcL, COLOR_BGR2GRAY);
    cuda::cvtColor(d_img2, d_srcR, COLOR_BGR2GRAY);

    Ptr<cuda::ORB> d_orb = cuda::ORB::create(500, 1.2f, 6, 31, 0, 2, 0, 31, 20, true);

    cuda::GpuMat d_keypointsL, d_keypointsR;
    cuda::GpuMat d_descriptorsL, d_descriptorsR;
    cuda::GpuMat d_descriptorsL_32F;
    cuda::GpuMat d_descriptorsR_32F;

    Ptr<cuda::DescriptorMatcher> d_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_L2);

    d_orb->detectAndComputeAsync(d_srcL, cuda::GpuMat(), d_keypointsL, d_descriptorsL);
    d_orb->convert(d_keypointsL, keypoints_1);
    d_descriptorsL.convertTo(d_descriptorsL_32F, CV_32F);

    d_orb->detectAndComputeAsync(d_srcR, cuda::GpuMat(), d_keypointsR, d_descriptorsR);
    d_orb->convert(d_keypointsR, keypoints_2);
    d_descriptorsR.convertTo(d_descriptorsR_32F, CV_32F);

#if KNN == 1
    std::vector<std::vector<DMatch>> knnMatches;
    std::vector<DMatch> good_matches;

    d_matcher->knnMatch(d_descriptorsL_32F, d_descriptorsR_32F, knnMatches, 2);

    const float ratio_thresh = 0.7f;

    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < ratio_thresh * knnMatches[i][1].distance) {
            good_matches.push_back(knnMatches[i][0]);
        }
    }

    for(size_t i = 0; i < good_matches.size(); i++){
        scense.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    for(unsigned int j = 0; j < scense.size(); j++){
        cv::circle(img_2, scense[j], 2, cv::Scalar(0, 255, 0), 2);
    }
    end = getTickCount();
    time = (double)(end-start)*1000/getTickFrequency();

    std::cout << "KNN time: " << time << " ms" << std::endl;

#elif KNN == 0
    std::vector<DMatch> matches;
    std::vector<DMatch> good_matches;

    d_matcher->match(d_descriptorsL_32F, d_descriptorsR_32F, matches);

    int sz = matches.size();

    double max_dist = 0;
    double min_dist = 100;

    for (int i = 0; i < sz; i++){
        double dist = matches[i].distance;
        if(dist < min_dist) {
            min_dist = dist;
        }
        if(dist > max_dist){
            max_dist = dist;
        }
    }
    cout << "\n-- Max dist: " << max_dist << endl;
    cout << "\n-- Min dist: " << min_dist << endl;

    for(int i = 0; i < sz; i++){
        if(matches[i].distance < 0.6*max_dist){
            good_matches.push_back(matches[i]);
        }
    }

    for(size_t i = 0; i < good_matches.size(); i++){
        scense.push_back(keypoints_2[good_matches[i].trainIdx].pt);
    }

    for(unsigned int j = 0; j < scense.size(); j++){
        cv::circle(img_2, scense[j], 2, cv::Scalar(0, 255, 0), 2);
    }
    end = getTickCount();
    time = (double)(end-start)*1000/getTickFrequency();

    std::cout << "Normal time: " << time << " ms" << std::endl;

#endif
    //-- Draw matches
    drawMatches( img_1, keypoints_1, img_2, keypoints_2, good_matches, imgMatches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    waitKey(0);

    return 0;
}
