#ifdef DLL_EXPORT
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
        #define DLL_API __declspec(dllexport)
    #else
        #define DLL_API
    #endif
#else
    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
        #define DLL_API __declspec(dllimport)
    #else
        #define DLL_API
    #endif
#endif

#ifndef LIBBG_MATTING_H
#define LIBBG_MATTING_H

#include <string>
#include <opencv2/opencv.hpp>
#include "background_matting.h"

using std::string;


extern "C" {

    DLL_API void BgMattingInit(int Width, int Height);
    DLL_API void BgMattingFinalize();
    DLL_API void BgMattingUseGpu(bool value);
    DLL_API void BgMattingLoadModel(char* ModelPathIn);
    DLL_API void BgMattingSetBackground(cv::Mat Image);
    DLL_API void BgMattingSetScene(cv::Mat Image);
    DLL_API cv::Mat BgMattingEstimateAlpha(cv::Mat Image);
    DLL_API cv::Mat BgMattingBlend(cv::Mat Image, cv::Mat Alpha, cv::Mat Scene);

}

#endif // LIBBG_MATTING_H