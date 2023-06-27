#pragma once

#include <cmath>
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
Vector2d InitVec2d(int Rows, int Cols);
Vector3d InitVec3d(int Nt, int Rows, int Cols);
Vector4d InitVec4d(int BatchSize, int Nt, int Rows, int Cols);

Vector2d NormalizeKeypoints(Vector2d& Keypoints, int FrameWidth, int FrameHeight);
Vector2d UnnormalizeKeypoints(Vector2d& Keypoints, int FrameWidth, int FrameHeight);
Vector2d NormalizeKeypoints3d(Vector2d& Keypoints, int FrameWidth, int FrameHeight);
Vector2d UnnormalizeKeypoints3d(Vector2d& Keypoints, int FrameWidth, int FrameHeight);

vector<float> InterpVec1d(vector<float>& InputVec, int OutputSize);

Vector4d ConvertKeypointsToInputVec(Vector2d& Keypoints, int BatchSize, int NumFrames);
Vector4d CreateInputVec(Vector3d& TemporalData, int BatchSize, int NumFrames);
torch::Tensor CreateInputTensor(Vector4d& InputVec);

Vector2d GetPoseFromOutputTensor(torch::Tensor& Outputs);

void GetPoseMinMax(float& Min, float& Max, Vector2d& PoseIn, int Direct);
Vector2d RescalePose2d(Vector2d& PoseIn, int FrameWidth, int FrameHeight);
Vector2d RescalePose3d(Vector2d& Pose3d, Vector2d& Pose2d);

Vector2d ToPixelSpace(Vector2d& PoseIn, int Width, int Height);

void PrintPoint(string Msg, vector<float> &Point, int Dims);
