#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"

int main(int argc, char **argv) {
    std::cout << "Hello, World!" << std::endl;

    if (argc != 3){
        std::cout << "usage: ORB_feature img1 img2" << std::endl;
        return 1;
    }

    cv::Mat img_1 = imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img_2 = imread(argv[2], cv::IMREAD_COLOR);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

    cv::Mat descriptors_1, descriptors_2;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // Detect oriented FAST concerns
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // Compute BRIEF descriptor
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    cv::Mat result_img_1;

    cv::drawKeypoints(img_1, keypoints_1, result_img_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB detected", result_img_1);

    // cv::waitKey(0);

    // Matcher BRIEF in 2 images by Hamming distance
    // Refer: https://en.wikipedia.org/wiki/Hamming_distance
    std::vector<cv::DMatch> matches;

    // Create BFMatcher matcher (NORM_HAMMING)
    matcher->match (descriptors_1, descriptors_2, matches);

    // Filter same points
    double min_dist = 10000, max_dist = 0;

    // Find the minimum and maximum distances between all matches,
    // that is, the distance between the most similar and least similar two sets of points
    for (int i = 0; i < descriptors_1.rows; i++){
        double dist = matches[i].distance;
        if(dist < min_dist){
            min_dist = dist;
        }
        if(dist > max_dist) {
            max_dist = dist;
        }
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // When the distance between the descriptors is greater than twice the minimum distance,
    // it is considered that the matching is wrong.
    // But sometimes the minimum distance will be very small,
    // and an empirical value of 30 is set as the lower limit.
    std::vector<cv::DMatch> good_matchers;
    for (int i = 0; i < descriptors_1.rows; i++){
        if(matches[i].distance <= cv::max(2*min_dist, 30.00)){
            good_matchers.push_back(matches[i]);
        }
    }

    // Draw maching result
    cv::Mat image_match;
    cv::Mat image_good_match;

    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, image_match);
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matchers, image_good_match);

    cv::imshow("Match Result", image_match);
    cv::imshow("Good Match Result", image_good_match);

    cv::waitKey(0);

    return 0;
}
