#include "mhformer.h"


MHFormer::MHFormer() {
}

MHFormer::~MHFormer() {
}

void MHFormer::Init(int FrameWidth, int FrameHeight) {

	mFrameWidth = FrameWidth;
	mFrameHeight = FrameHeight;

}

void MHFormer::UseGpu(bool bFlag) {

    mUseGpu = bFlag;  

    if (mUseGpu) {
        mDevice = torch::Device("cuda");
    } else {
        mDevice = torch::Device("cpu");
    }

}

bool MHFormer::LoadModel(string ModelPath) {

    bool bStatus = true;
    try {
        mModel = torch::jit::load(ModelPath);
        mModel.to(mDevice);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model. \n";
        bStatus = false;
    }

    return bStatus;

}

vector<vector<float>> MHFormer::Predict(vector<vector<float>>& Keypoints) {

    Vector2d pose2dPixel = Keypoints;
    pose2dPixel = RescaleAndShiftPose2d(pose2dPixel, mFrameWidth, mFrameHeight);

    // Normalize keypoints 2d
    Vector2d pose2d = NormalizeKeypoints(pose2dPixel, mFrameWidth, mFrameHeight);

    // Update temporal data
    mTemporalData.push_back(pose2d);
    if (mTemporalData.size() > mNumFramesUsed) {
        mTemporalData.erase(mTemporalData.begin());
    }

    Vector4d inputVec = CreateInputVec(mTemporalData, mBatchSize, mNumFramesModel);
    torch::Tensor inputTensor = CreateInputTensor(inputVec);

    // Inference
    torch::Tensor outputTensor = Infer(inputTensor);
    Vector2d pose3d = ConvertOutputTensorToPose3d(outputTensor);

    // Rotate pose around x-axis
    pose3d = RotatePose3dAroundX(pose3d, mAngleAroundX);

    // Unnormalize keypoints 3d
    Vector2d pose3dPixel = UnnormalizeKeypoints3d(pose3d, mFrameWidth, mFrameHeight);

    // Save unnormalized pose 3d
    mPose3dPixelUnnorm = pose3dPixel;

    // Rescale and rotate pose
    pose3dPixel = RescaleAndShiftPose3d(pose3dPixel, pose2dPixel);

    //float angleDeg = -10.0;
    //pose3dPixel = RotatePose3dAroundX(pose3dPixel, angleDeg);


    return pose3dPixel;

}

torch::Tensor MHFormer::Infer(torch::Tensor& Inputs) {

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(Inputs.to(mDevice));
    torch::Tensor outputs = mModel.forward(inputs).toTensor();
    outputs.to(torch::Device("cpu"));

    return outputs;

}

void MHFormer::SetAngleAroundX(float AngleDeg) {
    mAngleAroundX = AngleDeg;
}


Vector2d MHFormer::GetPose3dPixelUnnorm() {
    return mPose3dPixelUnnorm;
}