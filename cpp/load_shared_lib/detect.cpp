#include "bg_matting.hpp"
#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "timer.hpp"

using namespace std;
using namespace cv;

int main()
{
    Timer timer;

    float timeCost = 0.f;

    int width = 1280;
    int height = 720;
    //int width = 1920;
    //int height = 1080;

    int deviceId;

#ifdef __linux__
    deviceId = 0;
    VideoCapture capture(deviceId, CAP_V4L2);
#else
    deviceId = 1;
    VideoCapture capture(deviceId);
#endif

    capture.set(CAP_PROP_FRAME_WIDTH, width);
    capture.set(CAP_PROP_FRAME_HEIGHT, height);

    int result;
	result = (int)capture.get(CAP_PROP_AUTO_EXPOSURE);
	cout << "CAP_PROP_AUTO_EXPOSURE(original) = " << result << endl;
	result = (int)capture.get(CAP_PROP_EXPOSURE);
	cout << "CAP_PROP_EXPOSURE(original) = " << result << endl;
    //capture.set(CAP_PROP_AUTOFOCUS, 0); // Turn off auto focus
    capture.set(CAP_PROP_AUTO_EXPOSURE, 1); // 1: Turn 0ff, 3: Turn on auto exposure for OpenCV 4
    capture.set(CAP_PROP_EXPOSURE, 20); 
    //capture.set(CAP_PROP_EXPOSURE, 100); 

	result = (int)capture.get(CAP_PROP_AUTO_EXPOSURE);
	cout << "CAP_PROP_AUTO_EXPOSURE(after) = " << result << endl;
	result = (int)capture.get(CAP_PROP_EXPOSURE);
	cout << "CAP_PROP_EXPOSURE(after) = " << result << endl;

    if(!capture.isOpened()) {
        printf("Failed to open webcam. \n");
        return 1;
    }

    // Load the scene image
    Mat sceneImage = imread("sceneImage.png");
    resize(sceneImage, sceneImage, Size(width, height));

    // Shared library path
    string libPath = "/opt/lib/libbg_matting.so";

    // Load model
    //string modelPath = "torchscript_mobilenetv2_fp16.pth";
    string modelPath = "torchscript_mobilenetv2_fp32.pth";
    //string modelPath = "torchscript_resnet50_fp16.pth";
    //string modelPath = "torchscript_resnet50_fp32.pth";
    BgMatting matting = BgMatting();
    matting.LoadLibrary(libPath);
    matting.Init(width, height);

    // Use GPU or not
    matting.UseGpu(true);
    //matting.UseGpu(false);

    matting.LoadModel(modelPath); 

    // Set scene
    matting.SetScene(sceneImage);

    bool takeBackground = true;

    int keyCode = -1;
    Mat frame, inputImage, bgImage;
    Mat alpha;
    Mat blended;
    while (true) {

        // Read frame 
        if (!capture.read(frame))
        {
            printf("There is no frame available.");
            break;
        }

        keyCode = waitKey(1);

        // Show current frame
        imshow("image", frame);

        // Take a snapshot as the background
        if (char(keyCode) == 'b') {

            timer.delay(5000); // Wait for a while

            for (int k = 0; k < 10; k++) {
                capture.read(frame);
            }

            bgImage = frame.clone();

            imshow("bg", bgImage);
            imwrite("bgImage.png", bgImage);

            // Set background
            matting.SetBackground(bgImage);

            takeBackground = false;

        }

        // Image matting
        if (takeBackground == false) {


            // Input image 
            inputImage = frame.clone();
            imwrite("srcImage.png", inputImage);

            timer.tic();

            // Estimate alpha  
            alpha = matting.EstimateAlpha(inputImage);

            // Blend Image
            blended = matting.Blend(inputImage, alpha, sceneImage);

            timer.toc();

            timeCost = timer.get_dt();
            int fps = int(1.0 / timeCost);

            // Write message
            string msg;
            msg = "fps: " + to_string(fps);  
            int font = cv::FONT_HERSHEY_COMPLEX;
            putText(blended, msg, Point(30, 30), font, 1.0, Scalar(255, 255, 0), 2);

            imshow("alpha", alpha);
            imshow("blended", blended);

            imwrite("alpha.png", alpha);
            imwrite("blended.png", blended);
        }

        // Quit
        if (char(keyCode) == 'q') {
            break;
        }

    }

    // Finalization
    inputImage.release();
    bgImage.release();
    blended.release();
    capture.release();

    return 0;

}

