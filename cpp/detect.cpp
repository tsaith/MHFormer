#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <torch/script.h>

#include "libdetect.h"
#include "libplot.h"

using namespace std;
using namespace cv;

int main() {

    string msg;

    int frameWidth = 640;
    int frameHeight = 480;
    
    string modelPath = "../../checkpoint/pretrained/torchscript_model_traced.pth";
 
    const int batchSize= 1;
    const int numFramesModel= 81;
    const int numJoints = 17;
    const int dim2d = 2;
    const int dim3d = 3;


    Vector2d pose2d = GetMockKeypoints();
    cout << "pose2d: " << pose2d << endl;

    Vector4d mockInputVec = CreateMockInputVec(batchSize, numFramesModel, numJoints, dim2d);
    /*
    cout << "mockInputVec: " << mockInputVec << endl;
    cout << "mockInputVec x: " << mockInputVec[0][0][10][0] << endl;
    cout << "mockInputVec y: " << mockInputVec[0][0][10][1] << endl;
    */


    torch::Tensor inputTensor = CreateInputTensor(mockInputVec);


    //cout << "inputTensor: " << inputTensor << endl;

    cout << "Start to load the trained model." << endl;

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    //PrintTensorShape("inputTensor shape", inputTensor, 4);
    cout << "inputTensor.sizes():" << inputTensor.sizes() << endl;


    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    torch::Tensor outputTensor = model.forward(inputs).toTensor();

    //cout << "outputTensor[0][0]: " << outputTensor[0][0] << endl;

    Vector2d poseOut = GetPoseOut(outputTensor);

    Vector2d pose3d = RescalePose3d(poseOut, pose2d);
    cout << "pose3d: " << pose3d << endl;

    Vector2d pose3dPixel = ToPixelSpace(pose3d, frameWidth, frameHeight);
    //cout << "pose3dPixel: " << pose3dPixel << endl;

    Vector2d poseXY = GetPoseCrossSection(pose3dPixel, "x-y");
    Vector2d poseZY = GetPoseCrossSection(pose3dPixel, "z-y");
    cout << "poseZY: " << poseZY << endl;

    cv::Mat imageDiag = cv::Mat(frameHeight, frameWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    int shiftX, shiftY;

    /*
    shiftX = int(0.5*frameWidth);
    shiftY = int(0.5*frameHeight);
    PlotPose2dWithShift(imageDiag, poseXY, shiftX, shiftY);
    */

    shiftX = int(0.5*frameWidth);
    shiftY = int(0.5*frameHeight);
    PlotPose2dWithShift(imageDiag, poseZY, shiftX, shiftY);

    cv::imshow("win", imageDiag);

    int keyCode = cv::waitKey(0);

    cout << "End of run." << endl;

    return 0;

}

