#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

int main(int argc, const char* argv[]) {

    Mat targetImg = imread("C:\\Users\\user\\Desktop\\pic1.jpg");
    Mat sourceImg = imread("C:\\Users\\user\\Desktop\\pic2.jpg");

    //gray scale로 변환한다.
    Mat grayTarget1, graySource1;
    cvtColor(targetImg, grayTarget1, COLOR_BGR2GRAY);
    cvtColor(sourceImg, graySource1, COLOR_BGR2GRAY);


    // ORB descritor를 이용한다.
    Ptr<ORB> orb = ORB::create();

    vector<KeyPoint> featurePoints1, featurePoints2;
    Mat descriptors1, descriptors2;

    // ORB를 이용하여 영상의 특징점과 descriptor를 생성한다.
    orb->detectAndCompute(grayTarget1, noArray(), featurePoints1, descriptors1);
    orb->detectAndCompute(graySource1, noArray(), featurePoints2, descriptors2);

    // Brute-force matcher로 match 한다
    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING, true);
    matcher.match(descriptors1, descriptors2, matches);


    // match된 결과를 그릴 Mat선언
    Mat matchResultImg;

    sort(matches.begin(), matches.end());
    vector<DMatch> good_matches(matches.begin(), matches.begin() + 100);

    // match가 잘 됐는지 확인하기 위해 match 된 선을 그린다.
    drawMatches(targetImg, featurePoints1, sourceImg, featurePoints1, good_matches, matchResultImg);

    //imshow("matchResultImg", matchResultImg);


    vector<Point2f> points1, points2;
    for (int i = 0; i < good_matches.size(); i++) {
        points1.push_back(featurePoints1[good_matches[i].queryIdx].pt);
        points2.push_back(featurePoints2[good_matches[i].trainIdx].pt);
    }

    // match를 이용하여 homograpy를 구한다.
    Mat Homograpy = findHomography(points1, points2, RANSAC);
    //최종 stitching한 이미지를 저장할 Mat
    Mat result;

    // 위에서 계산한 Homograph를 이용하여 붙일 이미지를 warping한다.
    warpPerspective(sourceImg, result, Homograpy.inv(), Size(sourceImg.cols + targetImg.cols, sourceImg.rows + targetImg.rows));

    // warping한 이미지에 target 이미지를 카피해 붙인다.
    Mat concat(result, Rect(0, 0, targetImg.cols, targetImg.rows));
    targetImg.copyTo(concat);

    imshow("result_img", result);


    waitKey(0);

    return 0;

}
