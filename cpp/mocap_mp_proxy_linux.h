#pragma once

#include <stdio.h>
#include <iostream>
#include <string>
#include <filesystem> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef void* LibHandleT;

using namespace std;

class MocapMpProxy
{
public:

	MocapMpProxy();
	~MocapMpProxy();

    bool LoadLibrary(string filePath);
    void FreeLibrary();
    void ImportMethods();

    void Init(int width, int height);
    void Finalize();
    void Detect(cv::Mat image);
    void SetEngineName(string engineName);
    void Calibrate();
    const int GetNumBones();
    const int GetBoneDims();
    const int GetQuatDims();
    float* GetBone(int i);
    float* GetQuat(int i);

    const int GetMpPoseNumBones();
    float* GetMpPoseBone(int i);
    vector<vector<float>> GetMpPoseBones();

private:

    // Library handle 
    LibHandleT mHandle = NULL;

    // Define method type
    typedef void (*InitT)(int, int);
    typedef void (*FinalizeT)();
    typedef void (*SetEngineNameT)(char*);
    typedef void (*DetectT)(cv::Mat);
    typedef void (*CalibrateT)();

    typedef const int (*GetNumBonesT)();
    typedef const int (*GetBoneDimsT)();
    typedef const int (*GetQuatDimsT)();
    typedef float* (*GetBoneT)(int i);
    typedef float* (*GetQuatT)(int i);

    typedef const int (*GetMpPoseNumBonesT)();
    typedef float* (*GetMpPoseBoneT)(int i);

    InitT mInit = NULL;
    FinalizeT mFinalize = NULL;
    SetEngineNameT mSetEngineName = NULL;
    DetectT mDetect = NULL;
    CalibrateT mCalibrate = NULL;

    GetNumBonesT mGetNumBones = NULL;
    GetBoneDimsT mGetBoneDims = NULL;
    GetQuatDimsT mGetQuatDims = NULL;
    GetBoneT mGetBone = NULL;
    GetQuatT mGetQuat = NULL;

    GetMpPoseNumBonesT mGetMpPoseNumBones = NULL;
    GetMpPoseBoneT mGetMpPoseBone = NULL;

};