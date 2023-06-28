#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <torch/script.h>

#include "mhformer.h"
#include "mhformer_proxy.h"
#include "mhformer_utils.h"
#include "libplot.h"

using namespace std;
using namespace cv;


int main() {

    string msg;

    //int frameWidth = 640;
    //int frameHeight = 480;
    int frameWidth = 1280;
    int frameHeight = 720;
    
    const int videoLength = 15;
    string libPath = "../libs/libmhformer.so";
    string modelPath = "../../checkpoint/pretrained/torchscript_model_traced.pth";

    // Declare MHFormer
    MHFormerProxy mhformer;
    mhformer.LoadLibrary(libPath);
    mhformer.Init(frameWidth, frameHeight);
    mhformer.UseGpu(false);
    mhformer.LoadModel(modelPath);

    /*
    MHFormer mhformer;
    mhformer.Init(frameWidth, frameHeight);
    mhformer.UseGpu(false);
    mhformer.LoadModel(modelPath);
    */

    Vector2d pose2dPixel;
    Vector2d pose3dPixel;
    for (int i=0; i < videoLength; i++) {

        pose2dPixel = GetMockKeypoints();
        pose3dPixel = mhformer.Predict(pose2dPixel);

    }

    //cout << "pose3dPixel[0]: " << pose3dPixel[0] << endl;
    Vector2d poseXY = GetPoseCrossSection(pose3dPixel, "x-y");
    Vector2d poseZY = GetPoseCrossSection(pose3dPixel, "z-y");
    //cout << "poseXY: " << poseXY << endl;
    //cout << "poseZY: " << poseZY << endl;

    cv::Mat frontView = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat sideView = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat pose2dView = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat pose2dNormView = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    int shiftX, shiftY, shiftZ;
    shiftX = 0.5*frameWidth;
    shiftY = 0.5*frameHeight;
    shiftZ = 0.5*frameWidth;

    PlotPose2dWithShift(pose2dView, pose2dPixel, 0, 0);
    cv::imshow("pose2dView", pose2dView);

    PlotPose2dWithShift(frontView, poseXY, shiftX, shiftY);
    cv::imshow("frontView", frontView);

    PlotPose2dWithShift(sideView, poseZY, shiftZ, shiftY);
    cv::imshow("sideView", sideView);

    int keyCode = cv::waitKey(0);

    cout << "End of run." << endl;

    return 0;

}

