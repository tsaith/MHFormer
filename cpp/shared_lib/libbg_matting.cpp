#include "libbg_matting.h"


BackgroundMatting *pBgMatting = NULL;


void BgMattingInit(int Width, int Height) {
    pBgMatting = new BackgroundMatting(Width, Height);
}

void BgMattingFinalize() {

    if (!pBgMatting) {

        delete pBgMatting;
        pBgMatting = NULL;

    }

}
  
void BgMattingUseGpu(bool value) {
    pBgMatting->useGpu(value); 
}

void BgMattingLoadModel(char* ModelPathIn) {
	std::string ModelPath(ModelPathIn);
    pBgMatting->loadModel(ModelPath); 
}

void BgMattingSetBackground(cv::Mat Image) {
    pBgMatting->setBackground(Image);
}

void BgMattingSetScene(cv::Mat Image) {
    pBgMatting->setScene(Image);
}

cv::Mat BgMattingEstimateAlpha(cv::Mat Image) {
    cv::Mat alpha = pBgMatting->estimateAlpha(Image);
    return alpha;
}

cv::Mat BgMattingBlend(cv::Mat Image, cv::Mat Alpha, cv::Mat Scene) {
    cv::Mat blended = pBgMatting->blend(Image, Alpha, Scene);
    return blended;
}
