#include "libmhformer.h"
  
MHFormer* pInst = NULL;

const int numJoints = 17;
const int dim2d = 2;
const int dim3d = 3;

float Pose3dArray[numJoints][dim3d];

void MHFormerInit(int Width, int Height) {

    pInst = new MHFormer;
    pInst->Init(Width, Height);

}

void MHFormerFinalize() {

    if (!pInst) {
        delete pInst;
        pInst = NULL;
    }

}

void MHFormerUseGpu(bool bFlag) {
    pInst->UseGpu(bFlag); 
}

void MHFormerLoadModel(char* ModelPathIn) {

	std::string ModelPath(ModelPathIn);
    pInst->LoadModel(ModelPath); 

}

float* MHFormerPredict(float* KeypointsIn) {

    const int numJoints = 17;
    const int dim2d = 2;
    const int dim3d = 3;

    vector<vector<float>> pose2d(numJoints, vector<float>(dim2d, 0.0));
    vector<vector<float>> pose3d(numJoints, vector<float>(dim3d, 0.0));

    pose3d = pInst->Predict(pose2d);   

    float* pPose3d;



    return pPose3d;
}
