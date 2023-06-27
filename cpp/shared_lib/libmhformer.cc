#include "libmhformer.h"
#include "mhformer.h"
  
MHFormer *pMHFormer = NULL;

int MHFormerInit(int Width, int Height) {
    pMHFormer = new MHFormer;
    int Init_status = pMHFormer->Init(Width, Height);

    return Init_status;
}

void MHFormerFinalize() {
    if (!pMHFormer) {
        delete pMHFormer;
        pMHFormer = NULL;
    }
}

void MHFormerSetEngineName(char* Name) {

	std::string NameStr(Name);
    pMHFormer->SetEngineName(NameStr);

}

int MHFormerDetect(cv::Mat image) {

    int Detect_status = pMHFormer->Detect(image);

    return Detect_status;
}

void MHFormerCalibrate(void) {
    pMHFormer->Calibrate();
}

const int MHFormerGetNumBones() {
    return pMHFormer->GetNumBones();
}

const int MHFormerGetBoneDims() {
    return pMHFormer->GetBoneDims();
}

const int MHFormerGetQuatDims() {
    return pMHFormer->GetQuatDims();
}

float* MHFormerGetBone(int i) {
    return pMHFormer->GetBone(i);
}

float* MHFormerGetQuat(int i) {
    return pMHFormer->GetQuat(i);
}

const int MHFormerGetMpPoseNumBones() {
    return pMHFormer->GetMpPoseNumBones();
}

float* MHFormerGetMpPoseBone(int i) {
    return pMHFormer->GetMpPoseBone(i);
}
