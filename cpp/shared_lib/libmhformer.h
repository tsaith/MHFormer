#pragma once


#ifdef MHFORMER_EXPORTS

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

extern "C" {

    DLL_API int MHFormerInit(int Width, int Height);
    DLL_API void MHFormerFinalize(void);

}
