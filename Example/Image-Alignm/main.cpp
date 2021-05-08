#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

void alignImages(Mat &img1, Mat &img2, Mat &img1Reg, Mat &h){
    // COnvert images to graysacle
    Mat img1Gray, img2Gray;
    cvtColor(img1, img1Gray, cv::COLOR_BGR2GRAY);
    cvtColor(img2, img2Gray, cv::COLOR_BGR2GRAY);

    // Variables to store keypoints and descriptors
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    // Detec ORB faetures and compute descriptors
    Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
    orb->detectAndCompute(img1Gray, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2Gray, Mat(), keypoints2, descriptors2);

    // Match features
    std::vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors1, descriptors2, matches, Mat());

    // Sort matches by score
    std::sort(matches.begin(), matches.end());

    // Remove not so good matches
    const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Draw top matches
    Mat imgMatches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, imgMatches);
    imwrite("../images/matches.jpg", imgMatches);
    imshow("Good matches", imgMatches);

    // Extrac location of good matches
    std::vector<Point2f> points1, points2;

    for(size_t i = 0; i < matches.size(); i++){
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Find homography
    h = findHomography(points1, points2, RANSAC);

    // Use homography to warp image
    warpPerspective(img1, img1Reg, h, img2.size());
}
int main() {
    std::cout << "Hello, World!" << std::endl;

    // Read reference image
    string refFilename("../images/example-1.jpg");
    cout << "Reading reference image : " << refFilename << endl;
    Mat imReference = imread(refFilename);

    // Read image to be aligned
    string imFilename("../images/example-2.jpg");
    cout << "Reading image to align : " << imFilename << endl;
    Mat im = imread(imFilename);

    // Registered image will be resotred in imReg.
    // The estimated homography will be stored in h.
    Mat imReg, h;

    // Align images
    cout << "Aligning images ..." << endl;
    alignImages(im, imReference, imReg, h);

    // Write aligned image to disk.
    string outFilename("../images/aligned.jpg");
    cout << "Saving aligned image : " << outFilename << endl;
    imwrite(outFilename, imReg);
    imshow("Result", imReg);

    // Print estimated homography
    cout << "Estimated homography : \n" << h << endl;
    waitKey(0);
    return 0;
}

