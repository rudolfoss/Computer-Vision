//201821054 미디어학과 하태선 

#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

vector<Mat> GaussianPyramid(Mat img, int depth = 10) { //줄인이미지 저장
    vector<Mat> pyramid;
    pyramid.push_back(img);

    for (int i = 0; i < depth; i++) {
        Mat src;
        pyrDown(img, src);
        pyramid.push_back(src);
        img = src;
    }
    return pyramid;
}

vector<Mat> LaplacianPyramid(Mat img, int depth = 10) { 
    vector<Mat> pyramid;
    Mat src = img;

    for (int i = 0; i < depth; i++) {
        Mat small, tmp;
        pyrDown(src, small);
        pyrUp(small, tmp, src.size());
        pyramid.push_back(src - tmp);
        src = small;
    }
    pyramid.push_back(src);

    return pyramid;
}

Mat reconstruct(const vector<Mat>& pyramid){
    Mat ret = pyramid.back();    
    for (int i = pyramid.size()-2; i >= 0; i--) {
        pyrUp(ret, ret, pyramid[i].size());
        ret += pyramid[i];
    }
    return ret;
}

Mat blend(Mat img1, Mat img2, Mat mask) {
    Mat mul1, mul2, mask2, sum;
    multiply(img1, mask, mul1);
    mask2 = Scalar::all(1) - mask;
    multiply(img2, mask2, mul2);
    add(mul1, mul2, sum);
    return sum;
}


int main() {
    Mat image1 = imread("C:\\Users\\user\\Desktop\\apple.jpg");
    Mat image2 = imread("C:\\Users\\user\\Desktop\\orange.jpg");
    Mat mask = imread("C:\\Users\\user\\Desktop\\mask.png");
   
    vector<Mat> pyrmida;

    image1.convertTo(image1, CV_32F, 1 / 255.f);
    image2.convertTo(image2, CV_32F, 1 / 255.f);
    mask.convertTo(mask, CV_32F, 1 / 255.f);

    
    auto Lp_apple = LaplacianPyramid(image1); //라플 사과
    auto Lp_orange = LaplacianPyramid(image2); //라플 오렌지
    auto Ga_mask = GaussianPyramid(mask); //가우 마스크
 
    for (int i = 0; i <= 10; i++) {  // Mat과 vector<Mat> 형을 맞춰주기 위해 pushback
        auto num = blend(Lp_apple[i], Lp_orange[i], Ga_mask[i]);
        pyrmida.push_back(num);
    }

    Mat result = reconstruct(pyrmida);
    imshow("res", result);
    waitKey();

}