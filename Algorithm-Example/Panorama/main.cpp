#include <iostream>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv;

Stitcher::Mode mode = Stitcher::PANORAMA;

vector<Mat> imgs;

int main(int argc, char* argv[]) {

    std::cout << "Hello, World!" << std::endl;

    for (int i = 1; i < argc; i++){
        Mat img = imread(argv[i]);
        if (img.empty()){
            cout << "Can't read image" << argv[i] << "\n";
            return -1;
        }
        imgs.push_back(img);
    }
    Mat pano;

    Ptr<Stitcher> sticher = Stitcher::create(mode);

    Stitcher::Status status = sticher->stitch(imgs, pano);

    if(status != Stitcher::OK){
        cout << "Can't stch images \n";
        return -1;
    }
    imwrite("../resutl/result.jpg", pano);
    imshow("Result", pano);
    waitKey(0);

    return 0;
}
