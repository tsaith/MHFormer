#ifndef BG_MATTING_H
#define BG_MATTING_H

#include <opencv2/opencv.hpp>


using std::string;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
    #include <windows.h>
    typedef HINSTANCE LibHandleT;
    LibHandleT OpenSharedLibrary(string LibPath) {
        LibHandleT handle =  LoadLibrary(LibPath.c_str());
        if (handle == 0) {
            std::cout << "Cannot open the library." << std::endl;
        }

        return handle;
    }

    void CloseSharedLibrary(LibHandleT handle) {
        FreeLibrary(handle);
    }

    template<typename T>
    T GetFuncPointer(LibHandleT handle, string FuncName) {
        T fp = (T) GetProcAddress(handle, FuncName.c_str());
        if (!fp) {
            cerr << "Cannot load symbol "<< FuncName.c_str() << endl;
        }

        return fp;
    }

#else // For Linux

    #include <dlfcn.h>
    typedef void* LibHandleT;
    void* OpenSharedLibrary(string LibPath) {
        void *handle = dlopen(LibPath.c_str(), RTLD_LAZY);
        if (!handle) {
            std::cerr << "Cannot open the library: " << dlerror() << std::endl;
        }

        return handle;
    }

    void CloseSharedLibrary(void* handle) {
        dlclose(handle);
    }

    template<typename T>
    T GetFuncPointer(void* handle, string FuncName) {
        T fp = (T) dlsym(handle, FuncName.c_str());
        if (dlerror()) {
            std::cerr << "Cannot load symbol "<< FuncName.c_str() << std::endl;
        }

        return fp;
    }

#endif


class BgMatting {
public:

    BgMatting() {
    }

    ~BgMatting() {
        // Close the library
        CloseSharedLibrary(mHandle);
        mHandle = NULL;
    }

    void LoadLibrary(string LibPath) {
        // Load the shared library

        mHandle = OpenSharedLibrary(LibPath);

        // Define methods
        mInit = GetFuncPointer<InitT>(mHandle, "BgMattingInit");
        mFinalize = GetFuncPointer<FinalizeT>(mHandle, "BgMattingFinalize");
        mUseGpu = GetFuncPointer<UseGpuT>(mHandle, "BgMattingUseGpu");
        mLoadModel = GetFuncPointer<LoadModelT>(mHandle, "BgMattingLoadModel");
        mSetBackground = GetFuncPointer<SetBackgroundT>(mHandle, "BgMattingSetBackground");
        mSetScene = GetFuncPointer<SetSceneT>(mHandle, "BgMattingSetScene");
        mEstimateAlpha = GetFuncPointer<EstimateAlphaT>(mHandle, "BgMattingEstimateAlpha");
        mBlend = GetFuncPointer<BlendT>(mHandle, "BgMattingBlend");
    }


    void Init(int Width, int Height) {
        mInit(Width, Height);
    }

    void Finalize() {
        mFinalize();
    }

    void UseGpu(bool value) {
        mUseGpu(value);
    }

    void LoadModel(string ModelPath) {


        char* path = (char *)malloc(ModelPath.size() + 1);
        memcpy(path, ModelPath.c_str(), ModelPath.size() + 1); 

        mLoadModel(path);

        free(path);

    }

    void SetBackground(cv::Mat Image) {
        mSetBackground(Image);
    }

    void SetScene(cv::Mat Image) {
        mSetScene(Image);
    }

    cv::Mat EstimateAlpha(cv::Mat Image) {
        cv::Mat alpha = mEstimateAlpha(Image);
        return alpha;
    }

    cv::Mat Blend(cv::Mat Image, cv::Mat Alpha, cv::Mat Scene) {
        cv::Mat blended = mBlend(Image, Alpha, Scene);
        return blended;
    }


private:

    LibHandleT mHandle; // Hadle for library
    //HINSTANCE mHandle; // Hadle for library
 
    // Define function type
    typedef void (*InitT)(int, int);
    typedef void (*FinalizeT)();
    typedef void (*UseGpuT)(bool);
    typedef void (*LoadModelT)(char*);
    typedef void (*SetBackgroundT)(cv::Mat);
    typedef void (*SetSceneT)(cv::Mat);
    typedef cv::Mat (*EstimateAlphaT)(cv::Mat);
    typedef cv::Mat (*BlendT)(cv::Mat, cv::Mat, cv::Mat);

    InitT mInit; 
    FinalizeT mFinalize; 
    UseGpuT mUseGpu;
    LoadModelT mLoadModel; 
    SetBackgroundT mSetBackground;
    SetSceneT mSetScene;
    EstimateAlphaT mEstimateAlpha;
    BlendT mBlend;

};

#endif // BG_MATTING_H