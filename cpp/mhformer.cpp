#include "mhformer.h"


MHFormer::MHFormer(int FrameWidth, int FrameHeight) {

	mFrameWidth = FrameWidth;
	mFrameHeight = FrameHeight;

}

MHFormer::~MHFormer() {
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
    Vector2d pose2d = NormalizeKeypoints(pose2dPixel, mFrameWidth, mFrameHeight);

    // Update temporal data
    mTemporalData.push_back(pose2d);
    if (mTemporalData.size() > mNumFramesUsed) {
        mTemporalData.erase(mTemporalData.begin());
    }

    //Vector2d pose2dPixel = GetMockKeypoints();

    Vector4d inputVec = CreateInputVec(mTemporalData, mBatchSize, mNumFramesModel);
    torch::Tensor inputTensor = CreateInputTensor(inputVec);

    // Inference
    torch::Tensor outputTensor = Infer(inputTensor);

    Vector2d pose3d = GetPoseFromOutputTensor(outputTensor);

    Vector2d pose3dPixel = UnnormalizeKeypoints3d(pose3d, mFrameWidth, mFrameHeight);
    pose3dPixel = RescalePose3d(pose3dPixel, pose2dPixel);

    return pose3dPixel;

}

torch::Tensor MHFormer::Infer(torch::Tensor& Inputs) {

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(Inputs.to(mDevice));
    torch::Tensor outputs = mModel.forward(inputs).toTensor();
    outputs.to(torch::Device("cpu"));

    return outputs;

}
