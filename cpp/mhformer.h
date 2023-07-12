#pragma once

#include "mhformer_utils.h"

using namespace cv;
using namespace std;


class MHFormer
{

public:
    
    MHFormer();
    ~MHFormer();

    void Init(int FrameWidth, int FrameHeight);
    void UseGpu(bool bFlag);
    bool LoadModel(string ModelPath);

    vector<vector<float>> Predict(vector<vector<float>>& Pose2d);
    torch::Tensor Infer(torch::Tensor& Inputs);

    void SetAngleAroundX(float AngleDeg);
    Vector2d GetPose3dPixelUnnorm();


private:

    int mFrameWidth = 640;
    int mFrameHeight = 480;

    bool mUseGpu = false;
    torch::Device mDevice = torch::Device("cpu");
    //torch::Device mDevice = torch::Device("cuda");
    const c10::ScalarType mPrecision = torch::kFloat32;
    //const c10::ScalarType mPrecision = torch::kFloat16;

    torch::jit::script::Module mModel;

    const int mBatchSize= 1;
    const int mNumFramesUsed = 5;
    const int mNumFramesModel = 81;
    const int mNumJoints = 17;
    const int mDim2d = 2;
    const int mDim3d = 3;

    Vector3d mTemporalData;
    Vector4d mInputVec;

    Vector2d mPose3dPixelUnnorm;
    
    // Rotational angle around x-axis
    float mAngleAroundX = -10.0;

};

