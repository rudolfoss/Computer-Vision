#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {


    Mat img1 = imread("C:\\Users\\user\\Desktop\\pic1.jpg");
    Mat img2 = imread("C:\\Users\\user\\Desktop\\pic2.jpg");

    Mat greyImg1;
    Mat greyImg2;

    if (img1.empty() || img2.empty()) {
        return -1;
    }

    cvtColor(img1, greyImg1, COLOR_BGR2GRAY);
    cvtColor(img2, greyImg2, COLOR_BGR2GRAY);

    Ptr<FeatureDetector> detector = ORB::create();

    vector<KeyPoint> KeyPoint1, KeyPoint2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(greyImg1, noArray(), KeyPoint1, descriptors1);
    detector->detectAndCompute(greyImg2, noArray(), KeyPoint2, descriptors2);

    vector<DMatch> matches;
    BFMatcher matcher(NORM_HAMMING);
    matcher.match(descriptors1, descriptors2, matches);

    double dMaxDist = matches[0].distance;
    double dMinDist = matches[0].distance;
    double dDistance;

    for (int i = 0; i < descriptors1.rows; i++) {
        dDistance = matches[i].distance;

        if (dDistance < dMinDist) dMinDist = dDistance;
        if (dDistance > dMaxDist) dMaxDist = dDistance;
    }

    vector <DMatch> good_matches;
    int distance = 10;
    do {
        vector< DMatch> good_matches2;
        for (int i = 0; i < descriptors1.rows; i++) {
            if (matches[i].distance < distance * dMinDist)
                good_matches2.push_back(matches[i]);
        }
        good_matches = good_matches2;
        distance -= 1;
    } while (distance != 2 && good_matches.size() > 60);

    vector<Point2f> imgp1;
    vector<Point2f> imgp2;

    for (int i = 0; i < good_matches.size(); i++) {
        imgp1.push_back(KeyPoint1[good_matches[i].queryIdx].pt);
        imgp2.push_back(KeyPoint2[good_matches[i].trainIdx].pt);
    }

    Mat homoMatrix = findHomography(imgp1, imgp2, FM_RANSAC);

    Mat result;
    warpPerspective(img2, result, homoMatrix.inv(), Size(img2.cols + img1.cols, img2.rows + img1.rows));

    Mat comb(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(comb);

    imshow("result", result);
    waitKey(0);

    return 0;

}