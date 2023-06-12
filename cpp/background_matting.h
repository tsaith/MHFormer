#ifndef BACKGROUND_MATTING_H
#define BACKGROUND_MATTING_H

#include <torch/script.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


class BackgroundMatting
{
public:
    
    BackgroundMatting(int width, int height);
    ~BackgroundMatting();

    void useGpu(bool value);
    void loadModel(string modelPath);
    void setBackground(Mat bg);
    void setScene(Mat scene);

    void preprocess();
    Mat estimateAlpha(Mat img);
    Mat blend(Mat img, Mat alpha, Mat scene);
    Mat blendWithGray();

private:

    Mat normalize(Mat src);
    Mat unnormalize(Mat src);

    int mWidth = 0;
    int mHeight = 0;

    int mNumErosion = 0;
    int mNumDilation = 0;

    bool mUseGpu = true;
    torch::Device mDevice = torch::Device("cuda");
    const c10::ScalarType mPrecision = torch::kFloat32;
    //const c10::ScalarType mPrecision = torch::kFloat16;
    torch::jit::script::Module mModel;

    torch::Tensor mImgTensor;
    torch::Tensor mBgTensor;

    Mat mImg; 
    Mat mBg;
    Mat mScene;

    Mat mImageBlur, mBackgroundBlur;

    Mat mDiff;
    Mat mThresh;
    Mat mMask;
    Mat mTrimap;
    Mat mAlpha;
    Mat mBlended;

};

#endif
