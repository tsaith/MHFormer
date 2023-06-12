#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "mocap_mp_proxy_linux.h"
#include "timer.hpp"

using namespace std;
using namespace cv;

int main()
{

    MocapMpProxy mocap;

    Timer timer;

    float timeCost = 0.f;

    int width = 640;
    int height = 480;
    //int width = 1920;
    //int height = 1080;

    int cameraId;

    cameraId = 0;
    VideoCapture capture(cameraId, CAP_V4L2);

    capture.set(CAP_PROP_FRAME_WIDTH, width);
    capture.set(CAP_PROP_FRAME_HEIGHT, height);

    if(!capture.isOpened()) {
        printf("Failed to open webcam. \n");
        return 1;
    }

    string msg;
    // Mocap proxy
    bool loadLibStatus;
    string libPath = "../libs/libmocap_mp.so";
    loadLibStatus = mocap.LoadLibrary(libPath);
 
    if (!loadLibStatus) {
        msg = "Error: failed to load " + libPath + ".";
        cout << msg << endl;
    }


    mocap.Init(width, height);

    // Initialize Pose
    //const int poseNumBones = mocap.GetMpPoseNumBones(); 
    //const int dims = 3;
    //vector<vector<float>> poseBones(poseNumBones, vector<float> (dims, 0.0));
    vector<vector<float>> poseBones;

    // Load model
    string modelPath = "torchscript_mobilenetv2_fp32.pth";
    //matting.loadModel(modelPath); 

    int keyCode = -1;
    Mat frame, imageInput, imageDiag;
    Mat alpha;
    Mat blended;
    while (true) {

        // Read frame 
        if (!capture.read(frame))
        {
            printf("There is no frame available.");
            break;
        }

        // Input image 
        imageInput = frame.clone();
        imageDiag = frame.clone();

        timer.tic();
        mocap.Detect(imageInput);
        poseBones = mocap.GetMpPoseBones();
        timer.toc();

        timeCost = timer.get_dt();
        int fps = int(1.0 / timeCost);

        // Write message
        string msg;
        msg = "fps: " + to_string(fps);  
        int font = cv::FONT_HERSHEY_COMPLEX;
        putText(imageDiag, msg, Point(30, 30), font, 1.0, Scalar(255, 255, 0), 2);

        imshow("imageDiag", imageDiag);

        imwrite("imageDiag.jpg", imageDiag);

        // Quit
        keyCode = waitKey(1);
        if (char(keyCode) == 'q') {
            break;
        }

    }

    // Finalization
    capture.release();

    return 0;

}

