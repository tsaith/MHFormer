#include "mhformer.h"


MHFormer::MHFormer(int FrameWidth, int FrameHeight) {

	mFrameWidth = FrameWidth;
	mFrameHeight = FrameHeight;

    mDataVec = InitVec4d(mBatchSize, mNumFramesUsed, mNumJoints, mDim2d);
    mInputVec = InitVec4d(mBatchSize, mNumFramesModel, mNumJoints, mDim2d);
    
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
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model. \n";
        bStatus = false;
    }

    return bStatus;

}

Vector2d MHFormer::Predict(vector<vector<float>>& Keypoints) {

    mDataVec[0].push_back(Keypoints);
    mDataVec[0].erase(mDataVec[0].begin());
    
   return mKeypoints3d; 


}