#pragma once

#include "libdetect.h"

using namespace cv;
using namespace std;


class MHFormer
{
public:
    
    MHFormer(int FrameWidth, int FrameHeight);
    ~MHFormer();

    void UseGpu(bool bFlag);
    bool LoadModel(string ModelPath);

    vector<vector<float>> Predict(vector<vector<float>>& Keypoints);

private:

    int mFrameWidth = 0;
    int mFrameHeight = 0;

    bool mUseGpu = false;
    torch::Device mDevice = torch::Device("cuda");
    const c10::ScalarType mPrecision = torch::kFloat32;
    //const c10::ScalarType mPrecision = torch::kFloat16;

    torch::jit::script::Module mModel;

    const int mBatchSize= 1;
    const int mNumFramesUsed = 5;
    const int mNumFramesModel = 81;
    const int mNumJoints = 17;
    const int mDim2d = 2;
    const int mDim3d = 3;

    Vector2d mKeypoints;
    Vector4d mDataVec;
    Vector4d mInputVec;
    torch::Tensor mInputTensor;


    Vector2d mKeypoints3d;

};

