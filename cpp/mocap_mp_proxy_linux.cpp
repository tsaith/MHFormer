#include "mocap_mp_proxy_linux.h"
#include <dlfcn.h>

LibHandleT OpenSharedLibrary(string LibPath) {

    LibHandleT handle = dlopen(LibPath.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot open the library: " << dlerror() << std::endl;
    }

    return handle;
}

void CloseSharedLibrary(LibHandleT handle) {
    dlclose(handle);
}

template<typename T>
T GetFuncPointer(LibHandleT handle, string FuncName) {
    T fp = (T) dlsym(handle, FuncName.c_str());
    if (dlerror()) {
        std::cerr << "Cannot load symbol "<< FuncName.c_str() << std::endl;
    }

    return fp;
}

MocapMpProxy::MocapMpProxy() {
}

MocapMpProxy::~MocapMpProxy() {
    FreeLibrary();
} 

bool MocapMpProxy::LoadLibrary(string LibPath) {

    bool exist = std::filesystem::exists(LibPath);
    bool status = true;

    if (exist) {

        mHandle = OpenSharedLibrary(LibPath);;
        ImportMethods();

    } else {
        status = false;
    }

    return status;

}

void MocapMpProxy::FreeLibrary() {

    if (mHandle != NULL)
    {
        CloseSharedLibrary(mHandle);
        mHandle = NULL;
    }

}

void MocapMpProxy::ImportMethods() {

    if (mHandle != NULL)
    {
        mInit = GetFuncPointer<InitT>(mHandle, "MocapMpInit");
        mFinalize = GetFuncPointer<FinalizeT>(mHandle, "MocapMpFinalize");
        mSetEngineName = GetFuncPointer<SetEngineNameT>(mHandle, "MocapMpSetEngineName");
        mDetect = GetFuncPointer<DetectT>(mHandle, "MocapMpDetect");
        mCalibrate = GetFuncPointer<CalibrateT>(mHandle, "MocapMpCalibrate");
        mGetNumBones = GetFuncPointer<GetNumBonesT>(mHandle, "MocapMpGetNumBones");
        mGetBoneDims = GetFuncPointer<GetBoneDimsT>(mHandle, "MocapMpGetBoneDims");
        mGetQuatDims = GetFuncPointer<GetQuatDimsT>(mHandle, "MocapMpGetQuatDims");
        mGetBone = GetFuncPointer<GetBoneT>(mHandle, "MocapMpGetBone");
        mGetQuat = GetFuncPointer<GetQuatT>(mHandle, "MocapMpGetQuat");

        mGetMpPoseNumBones = GetFuncPointer<GetMpPoseNumBonesT>(mHandle, "MocapMpGetMpPoseNumBones");
        mGetMpPoseBone = GetFuncPointer<GetMpPoseBoneT>(mHandle, "MocapMpGetMpPoseBone");

    }

}


void MocapMpProxy::Init(int width, int height) {

    if (mInit != NULL) {
        mInit(width, height);
    }
    else {
        cout << "Error: mInit is NULL." << endl;
    }

}

void MocapMpProxy::Finalize() {

    if (mFinalize != NULL) {
        mFinalize();
    }
    else {
        cout << "Error: mFinalize is NULL." << endl;
    }

}

void MocapMpProxy::SetEngineName(string engineName) {

    if (mSetEngineName != NULL) {
        char* name = (char*) engineName.c_str();
        mSetEngineName(name);
    }
    else {
        cout << "Error: mSetEngineName is NULL." << endl;
    }

}

void MocapMpProxy::Detect(cv::Mat image) {

    if (mDetect != NULL) {
        mDetect(image);
    }
    else {
        cout << "Error: mDetect is NULL." << endl;
    }

}

void MocapMpProxy::Calibrate() {

    if (mCalibrate != NULL) {
        mCalibrate();
    }
    else {
        cout << "Error: mCalibrate is NULL." << endl;
    }

}

const int MocapMpProxy::GetNumBones() {

    if (mGetNumBones != NULL) {
        return mGetNumBones();
    }
    else {
        cout << "Error: mGetNumBones is NULL." << endl;
        return -1;
    }

}

const int MocapMpProxy::GetBoneDims() {

    if (mGetBoneDims != NULL) {
        return mGetBoneDims();
    }
    else {
        cout << "Error: mGetBoneDims is NULL." << endl;
        return -1;
    }

}

const int MocapMpProxy::GetQuatDims() {

    if (mGetQuatDims != NULL) {
        return mGetQuatDims();
    }
    else {
        cout << "Error: mGetQuatDims is NULL." << endl;
        return -1;
    }

}

float* MocapMpProxy::GetBone(int i) {

    float* bone;
    if (mGetBone != NULL) {
        bone = mGetBone(i);
    }
    else {
        cout << "Error: mGetBone is NULL." << endl;
        bone = NULL;
    }

    return bone;
}

float* MocapMpProxy::GetQuat(int i) {

    float* quat;
    if (mGetQuat != NULL) {
         quat = mGetQuat(i);
    }
    else {
        cout << "Error: mGetQuat is NULL." << endl;
        quat = NULL;
    }

    return quat;

}

float* MocapMpProxy::GetMpPoseBone(int i) {

    float* bone;
    if (mGetMpPoseBone != NULL) {
        bone = mGetMpPoseBone(i);
    }
    else {
        cout << "Error: mGetMpPoseBone is NULL." << endl;
        bone = NULL;
    }

    return bone;
}

const int MocapMpProxy::GetMpPoseNumBones() {

    if (mGetMpPoseNumBones != NULL) {
        return mGetMpPoseNumBones();
    }
    else {
        cout << "Error: mGetMpPoseNumBones is NULL." << endl;
        return -1;
    }

}

vector<vector<float>> MocapMpProxy::GetMpPoseBones() {

    const int num = GetMpPoseNumBones(); 
    const int dims = 3;
    vector<vector<float>> bones(num, vector<float> (dims, 0.0));
    
    float* pBone;
    for (int i; i < num; i++) {
        pBone = GetMpPoseBone(i);
        for (int j; j < dims; j++) {
            bones[i][j] = pBone[j];
        }
    }

    return bones;

}