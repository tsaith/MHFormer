#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <torch/script.h>

#include "mhformer.h"
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
    
    string modelPath = "../../checkpoint/pretrained/torchscript_model_traced.pth";
 
    const int batchSize= 1;
    const int numFramesModel= 81;
    const int numJoints = 17;
    const int dim2d = 2;
    const int dim3d = 3;

    Vector2d pose2dPixel = GetMockKeypoints();
    //Vector2d pose2d = GetMockKeypointsTmp();
    //cout << "pose2d: " << pose2d << endl;

    Vector2d pose2d = NormalizeKeypoints(pose2dPixel, frameWidth, frameHeight);
    Vector4d inputVec = ConvertKeypointsToInputVec(pose2d, batchSize, numFramesModel);

    torch::Tensor inputTensor = CreateInputTensor(inputVec);
    //cout << "inputTensor: " << inputTensor << endl;

    cout << "Start to load the trained model." << endl;

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model. \n";
        return -1;
    }

    //PrintTensorShape("inputTensor shape", inputTensor, 4);
    //cout << "inputTensor.sizes():" << inputTensor.sizes() << endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    torch::Tensor outputTensor = model.forward(inputs).toTensor();

    Vector2d pose3d;
    pose3d = GetPoseFromOutputTensor(outputTensor);

    //pose3d = RescalePose3d(pose3d, pose2d);
    //cout << "pose3d: " << pose3d << endl;

    //Vector2d pose3dPixel = ToPixelSpace(pose3d, frameWidth, frameHeight);
    Vector2d pose3dPixel = UnnormalizeKeypoints3d(pose3d, frameWidth, frameHeight);
    //cout << "pose3dPixel: " << pose3dPixel << endl;
    pose3dPixel = RescalePose3d(pose3dPixel, pose2dPixel);

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

    //PlotPose2dWithShift(pose2dNormView, pose2dNorm, shiftX, shiftY);
    //cv::imshow("pose2dNormView", pose2dNormView);

    PlotPose2dWithShift(frontView, poseXY, shiftX, shiftY);
    cv::imshow("frontView", frontView);

    PlotPose2dWithShift(sideView, poseZY, shiftZ, shiftY);
    cv::imshow("sideView", sideView);

    int keyCode = cv::waitKey(0);

    cout << "End of run." << endl;

    return 0;

}

