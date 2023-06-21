#pragma once

#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <torch/script.h>

using namespace std;
using namespace cv;

typedef vector<vector<float>> Vector2d;
typedef vector<Vector2d> Vector3d;
typedef vector<Vector3d> Vector4d;

Vector2d GetMockKeypoints();
Vector2d GetMockKeypointsFlip();
Vector2d GetMockOutputs();
Vector2d GetMockOutputsFlip();
Vector2d InitVec2d(int Rows, int Cols);
Vector3d InitVec3d(int Nt, int Rows, int Cols);
Vector4d InitVec4d(int BatchSize, int Nt, int Rows, int Cols);

Vector4d CreateMockInputVec(int BatchSize, int NumFrames, int NumJoints, int Dim2d);
torch::Tensor CreateInputTensor(Vector4d& InputVec);
Vector2d GetPoseOut(at::Tensor& Outputs);

void GetPoseMinMax(float& Min, float& Max, Vector2d& PoseIn, int Direct);
Vector2d RescalePose3d(Vector2d& Pose3d, Vector2d& Pose2d);

Vector2d ToPixelSpace(Vector2d& PoseIn, int Width, int Height);

void PrintPoint(string Msg, vector<float> &Point, int Dims);