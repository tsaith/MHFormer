#include <torch/script.h>

#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "mhformer_proxy.h"
#include "mhformer_utils.h"
#include "libplot.h"
#include "timer.hpp"

using namespace std;
using namespace cv;


int main() {

    string msg;

    //int frameWidth = 640;
    //int frameHeight = 480;
    int frameWidth = 1280;
    int frameHeight = 720;
    
    const int videoLength = 15;
    string libPath = "../../libs/libmhformer.so";
    string modelPath = "../../checkpoint/pretrained/torchscript_model_traced.pth";

    // Timer
    Timer timer;

    // Declare MHFormer
    MHFormerProxy mhformer;
    mhformer.LoadLibrary(libPath);
    mhformer.Init(frameWidth, frameHeight);
    //mhformer.UseGpu(false);
    mhformer.UseGpu(true);
    mhformer.LoadModel(modelPath);

    /*
    MHFormer mhformer;
    mhformer.Init(frameWidth, frameHeight);
    mhformer.UseGpu(false);
    mhformer.LoadModel(modelPath);
    */

    Vector2d pose2dPixel;
    Vector2d pose3dPixel;

    // Warm-up of inference
    for (int i=0; i < 3; i++) {
        pose2dPixel = GetMockKeypoints();
        pose3dPixel = mhformer.Predict(pose2dPixel);
    }

    timer.tic();
    for (int i=0; i < videoLength; i++) {
        pose2dPixel = GetMockKeypoints();
        pose3dPixel = mhformer.Predict(pose2dPixel);
    }
    timer.toc();

    float timeCostTot = timer.GetDt();
    float timeCost = timeCostTot / videoLength;
    cout << "time cost of prediction is " << timeCost << " sec." << endl;

    // Cross-section
    Vector2d poseXY = GetPoseCrossSection(pose3dPixel, "x-y");
    Vector2d poseZY = GetPoseCrossSection(pose3dPixel, "z-y");

    // Make plots
    cv::Mat frontView = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat sideView = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat pose2dView = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat pose2dNormView = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    PlotPose2d(pose2dView, pose2dPixel);
    cv::imshow("pose2dView", pose2dView);

    PlotPose2d(frontView, poseXY);
    cv::imshow("frontView", frontView);

    PlotPose2d(sideView, poseZY);
    cv::imshow("sideView", sideView);

    int keyCode = cv::waitKey(0);

    cout << "End of run." << endl;

    return 0;

}

