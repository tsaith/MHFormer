#include "background_matting.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <torch/script.h>


using namespace cv;
using namespace std;

string get_image_type(const Mat& img, bool more_info=true) 
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
   
    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    cout << "r: " << r << endl;

    return r;
}

void show_image(Mat& img, string title)
{
    std::string image_type = get_image_type(img);

    namedWindow(title + " type:" + image_type, cv::WINDOW_NORMAL); // Create a window for display.
    imshow(title, img);
    waitKey(0);
}

auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 })
{
    tensor = tensor.permute(dims);
    return tensor;
}


auto toTensorImage(Mat img, torch::Device device, c10::ScalarType precision)
{
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);
    tensor_image = tensor_image.to(device).to(precision);
    tensor_image = tensor_image.unsqueeze_(0).mul(1.0/255);
    tensor_image = transpose(tensor_image);

    return tensor_image;
}

Mat toCvImage(at::Tensor tensor)
{
    int channels = tensor.sizes()[1];
    int height = tensor.sizes()[2];
    int width = tensor.sizes()[3];

    Mat output;

    if (channels == 1) {
        output = Mat(height, width, CV_8UC1);
        tensor = tensor.squeeze().detach().permute({0, 1});
    } else {
        output = Mat(height, width, CV_8UC3);
        tensor = tensor.squeeze().detach().permute({1, 2, 0});
    } 

    tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
    tensor = tensor.to(torch::kCPU);

    std::memcpy((void *) output.data, tensor.data_ptr(), sizeof(torch::kU8) * tensor.numel());

    return output;

}


BackgroundMatting::BackgroundMatting(int width, int height) {

	mWidth = width;
	mHeight = height;

}

BackgroundMatting::~BackgroundMatting() {

}

void BackgroundMatting::useGpu(bool value) {
    mUseGpu = value;  

    if (mUseGpu) {
        mDevice = torch::Device("cuda");
    } else {
        mDevice = torch::Device("cpu");
    }

}

void BackgroundMatting::loadModel(string modelPath) {

    mModel = torch::jit::load(modelPath);
    mModel.setattr("backbone_scale", 0.25);
    mModel.setattr("refine_mode", "sampling");
    mModel.setattr("refine_sample_pixels", 80000);
    mModel.to(mDevice);

}

void BackgroundMatting::setBackground(Mat bg) {

    mBg = bg;
    mBgTensor = toTensorImage(mBg, mDevice, mPrecision);

}

void BackgroundMatting::setScene(Mat scene) {
    mScene = scene;
}

void BackgroundMatting::preprocess() {
}


Mat BackgroundMatting::estimateAlpha(Mat img) {

    //auto src = torch::rand({1, 3, 1080, 1920}).to(mDevice).to(mPrecision);
    //auto bgr = torch::rand({1, 3, 1080, 1920}).to(mDevice).to(mPrecision);

    mImg = img; 
    mImgTensor = toTensorImage(mImg, mDevice, mPrecision);
    auto outputs = mModel.forward({mImgTensor, mBgTensor}).toTuple()->elements();

    auto alphaTensor = outputs[0].toTensor();
    auto fgTensor = outputs[1].toTensor();

    mAlpha = toCvImage(alphaTensor);

    int rows = mAlpha.rows;
    int cols = mAlpha.cols;

    return mAlpha;
}


Mat BackgroundMatting::normalize(Mat src)
{
    Mat out;
    src.convertTo(out, CV_32F, 1.0 / 255, 0);
    return out;
}

Mat BackgroundMatting::unnormalize(Mat src)
{
    Mat out;
    src.convertTo(out, CV_8U, 255, 0);
    return out;
}

Mat BackgroundMatting::blend(Mat imgIn, Mat alphaIn, Mat sceneIn)
{
    Mat img = normalize(imgIn);
    Mat alphaC3;
    cvtColor(alphaIn, alphaC3, COLOR_GRAY2BGR);
    Mat alpha = normalize(alphaC3);
    Mat scene = normalize(sceneIn);
    Mat tmp1, tmp2;

    cv::Scalar ones = Scalar(1.0, 1.0, 1.0);
    multiply(alpha, img, tmp1, 1.0);
    multiply((ones - alpha), scene, tmp2, 1.0);
    Mat blended = tmp1 + tmp2;

    Mat out = unnormalize(blended);

    return out;

};

Mat BackgroundMatting::blendWithGray()
{
    Mat bg(mImg.size(), CV_8UC3, Scalar(127, 127, 127));
    Mat out = blend(mImg, mAlpha, bg);
    return out;
}
