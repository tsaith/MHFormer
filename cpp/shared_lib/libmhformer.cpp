#include "libmhformer.h"
  
MHFormer* pInst = NULL;

const int NumJoints = 17;
const int Dim3d = 3;

float Pose3dArray[NumJoints][Dim3d];

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

void MHFormerLoadModel(char* ModelPath) {

	std::string ModelPathStr(ModelPath);
    pInst->LoadModel(ModelPathStr); 

}

float* MHFormerPredict(float* pKeypoints) {

    const int numJoints = 17;
    const int dim2d = 2;

    vector<vector<float>> pose2d(numJoints, vector<float>(dim2d, 0.0));
    for (int i=0; i < numJoints; i++) {
        for (int j=0; j < dim2d; j++) {
            pose2d[i][j] = *(pKeypoints + i*dim2d + j);
        }
    }

    vector<vector<float>> pose3d;
    pose3d = pInst->Predict(pose2d);

    const int dim3d = 3;
    for (int i=0; i < numJoints; i++) {
        for (int j=0; j < dim3d; j++) {
            Pose3dArray[i][j] = pose3d[i][j];
        }
    }

    float* pPose3d = &Pose3dArray[0][0];

    return pPose3d;

}
