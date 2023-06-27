#include "mhformer_utils.h"


Vector2d GetMockKeypoints() {

    Vector2d keypoints {

        {640.0000, 360.0000},
        {605.3012, 349.5903},
        {598.3614, 498.7952},
        {501.2048, 526.5542},
        {674.6988, 370.4096},
        {657.3494, 540.4338},
        {660.8193, 696.5783},
        {662.5542, 273.2530},
        {643.4699, 180.7229},
        {626.1205, 155.2771},
        {623.5180, 120.5783},
        {716.3374, 186.5060},
        {771.8554, 252.4337},
        {709.3976, 294.0723},
        {598.3614, 186.5060},
        {539.3735, 231.6144},
        {598.3614, 294.0723}

    };

    return keypoints;

}

Vector2d InitVec2d(int Rows, int Cols){
    Vector2d vec2d(Rows, vector<float>(Cols, 0.0));
    return vec2d;
}

Vector3d InitVec3d(int Nt, int Rows, int Cols){

    Vector3d vec3d;
    for (int i=0; i < Nt; i++) {
        vec3d.push_back(InitVec2d(Rows, Cols));
    }

    return vec3d;

}

Vector4d InitVec4d(int BatchSize, int Nt, int Rows, int Cols){

    Vector4d vec4d;
    for (int i=0; i < BatchSize; i++) {
        vec4d.push_back(InitVec3d(Nt, Rows, Cols));
    }

    return vec4d;

}


Vector2d NormalizeKeypoints(Vector2d& Keypoints, int FrameWidth, int FrameHeight) {

    Vector2d keypoints = Keypoints;

    int numJoints = Keypoints.size();
    int dim2d = Keypoints[0].size();

    for (int i=0; i < numJoints; i++) {

        keypoints[i][0] = 2.0*keypoints[i][0]/FrameWidth - 1.0;
        keypoints[i][1] = 2.0*keypoints[i][1]/FrameWidth - 1.0*FrameHeight/FrameWidth;

    }

    return keypoints;

}

Vector2d UnnormalizeKeypoints(Vector2d& Keypoints, int FrameWidth, int FrameHeight) {

    Vector2d keypoints = Keypoints;

    int numJoints = Keypoints.size();
    int dim2d = Keypoints[0].size();

    for (int i=0; i < numJoints; i++) {

        keypoints[i][0] = (keypoints[i][0] + 1.0)*0.5*FrameWidth;
        keypoints[i][1] = (keypoints[i][1] + 1.0*FrameHeight/FrameWidth)*0.5*FrameWidth;

    }

    return keypoints;

}

Vector2d NormalizeKeypoints3d(Vector2d& Keypoints, int FrameWidth, int FrameHeight) {

    Vector2d keypoints = Keypoints;

    int numJoints = Keypoints.size();
    for (int i=0; i < numJoints; i++) {

        keypoints[i][0] = 2.0*keypoints[i][0]/FrameWidth - 1.0;
        keypoints[i][1] = 2.0*keypoints[i][1]/FrameWidth - 1.0*FrameHeight/FrameWidth;
        keypoints[i][2] = 2.0*keypoints[i][2]/FrameWidth - 1.0;

    }

    return keypoints;

}

Vector2d UnnormalizeKeypoints3d(Vector2d& Keypoints, int FrameWidth, int FrameHeight) {

    Vector2d keypoints = Keypoints;

    int numJoints = Keypoints.size();
    for (int i=0; i < numJoints; i++) {

        keypoints[i][0] = (keypoints[i][0] + 1.0)*0.5*FrameWidth;
        keypoints[i][1] = (keypoints[i][1] + 1.0*FrameHeight/FrameWidth)*0.5*FrameWidth;
        keypoints[i][2] = (keypoints[i][2] + 1.0)*0.5*FrameWidth;

    }

    return keypoints;

}

vector<float> InterpVec1d(vector<float>& InputVec, int OutputSize) {

    int inputSize = InputVec.size();
    std::vector<float> outputVec(OutputSize);

    for (int i = 0; i < OutputSize; ++i) {

        float index = static_cast<float>(i * (inputSize - 1)) / static_cast<float>(OutputSize - 1);
        int index_int = static_cast<int>(index);
        float t = index - index_int;

        outputVec[i] = (1.0f - t) * InputVec[index_int] + t * InputVec[std::min(index_int + 1, inputSize - 1)];
    }

    return outputVec;
}

void InterpInputVec(Vector2d& InputVec, int NumFramesOut) {



}


Vector4d ConvertKeypointsToInputVec(Vector2d& Keypoints, int BatchSize, int NumFrames) {

    Vector2d keypoints = Keypoints;

    int numJoints = Keypoints.size();
    int dim2d = Keypoints[0].size();

    Vector4d inputVec = InitVec4d(BatchSize, NumFrames, numJoints, dim2d);

    for (int ib=0; ib < BatchSize; ib++) {
        for (int i=0; i < NumFrames; i++) {
            for (int j=0; j < numJoints; j++) {
                for (int k=0; k < dim2d; k++) {
                    inputVec[ib][i][j][k] = Keypoints[j][k];
                }
            }
        }
    }

    return inputVec;

}

Vector4d CreateInputVec(Vector3d& TemporalData, int BatchSize, int NumFramesOut) {

    int numFramesIn = TemporalData.size();
    int numJoints = TemporalData[0].size();
    int dim2d = TemporalData[0][0].size();

    Vector4d inputVec = InitVec4d(BatchSize, NumFramesOut, numJoints, dim2d);

    Vector3d data = InitVec3d(numJoints, dim2d, numFramesIn);
    Vector3d dataInterp = InitVec3d(numJoints, dim2d, NumFramesOut);

    // Prepare data for interpolation
    for (int i=0; i < numFramesIn; i++) {
        for (int j=0; j < numJoints; j++) {
            for (int k=0; k < dim2d; k++) {
                data[j][k][i] = TemporalData[i][j][k];
            }
        }
    }

    for (int j=0; j < numJoints; j++) {
        for (int k=0; k < dim2d; k++) {
            dataInterp[j][k] = InterpVec1d(data[j][k], NumFramesOut);
        }
    }

    // Input vector
    for (int ib=0; ib < BatchSize; ib++) {
        for (int i=0; i < NumFramesOut; i++) {
            for (int j=0; j < numJoints; j++) {
                for (int k=0; k < dim2d; k++) {
                    inputVec[ib][i][j][k] = dataInterp[j][k][i];
                }
            }
        }
    }

    return inputVec;

}

torch::Tensor CreateInputTensor(Vector4d& InputVec) {

    int batchSize = InputVec.size();
    int numFrames = InputVec[0].size();
    int numJoints = InputVec[0][0].size();
    int dim2d = InputVec[0][0][0].size();

    torch::Tensor inputTensor = torch::zeros({batchSize, numFrames, numJoints, dim2d},
        torch::kFloat);

    for (int ib=0; ib < batchSize; ib++) {
        for (int i=0; i < numFrames; i++) {
            for (int j=0; j < numJoints; j++) {
                for (int k=0; k < dim2d; k++) {
                    inputTensor[ib][i][j][k] = InputVec[ib][i][j][k];
                }
            }
        }
    }

    return inputTensor;

}

Vector2d ConvertOutputTensorToPose3d(torch::Tensor& Outputs) {

    int numJoints = Outputs.size(2);
    int dims = 3;

    Vector2d pose = InitVec2d(numJoints, dims);
    for (int i=0; i < numJoints; i++) {
        for (int j=0; j < dims; j++) {
            pose[i][j] = Outputs[0][0][i][j].item<float>();
        }
    }

    return pose;

}

void GetPoseMinMax(float& Min, float& Max, Vector2d& PoseIn, int Direct) {

    int numJoints = PoseIn.size();

    vector<float> data;
    for (int i=0; i < numJoints; i++) {
        data.push_back(PoseIn[i][Direct]);
    }

    auto it = std::minmax_element(data.begin(), data.end());
    Min = *it.first;
    Max = *it.second;
 
}

Vector2d RescalePose2d(Vector2d& PoseIn, int FrameWidth, int FrameHeight) {

    Vector2d pose = PoseIn;

    int numJoints = pose.size();
    int dims = pose[0].size();

    float xMin, xMax;
    float yMin, yMax;

    // Bounds of pose 2d
    GetPoseMinMax(xMin, xMax, pose, 0);
    GetPoseMinMax(yMin, yMax, pose, 1);

    float humanWidth = abs(xMax - xMin);
    float humanHeight = abs(yMax - yMin);

    float resizeFactor = 0.8;
    float errTol = 1.0e-8;
    float xRatio, yRatio, targetRatio;

    if (humanWidth > errTol && humanHeight > errTol) {

        xRatio = 1.0 * FrameWidth / humanWidth;
        yRatio = 1.0 * FrameHeight / humanHeight;
        targetRatio = resizeFactor*min(xRatio, yRatio);

        for (int i=0; i < numJoints; i++) {
            for (int d=0; d < dims; d++) {
                pose[i][d] *= targetRatio;
            }
        }

    } else {
        return pose;
    }

    // Set the pelvis as center
    float x0 = 0.5*FrameWidth;
    float y0 = 0.5*FrameHeight;

    vector<float> pelvis = pose[0];    
    float xShift = x0 - pelvis[0]; 
    float yShift = y0 - pelvis[1]; 

    for (int i=0; i < numJoints; i++) {
        pose[i][0] += xShift;
        pose[i][1] += yShift;
    }

    return pose;

}

Vector2d RescalePose3d(Vector2d& Pose3d, Vector2d& Pose2d) {

    Vector2d pose3d = Pose3d;

    int numJoints = pose3d.size();
    int dims = 3;

    float xMinP2d, xMaxP2d;
    float yMinP2d, yMaxP2d;

    float xMinP3d, xMaxP3d;
    float yMinP3d, yMaxP3d;

    // Bounds of pose 2d
    GetPoseMinMax(xMinP2d, xMaxP2d, Pose2d, 0);
    GetPoseMinMax(yMinP2d, yMaxP2d, Pose2d, 1);

    // Bounds of pose 3d
    GetPoseMinMax(xMinP3d, xMaxP3d, Pose3d, 0);
    GetPoseMinMax(yMinP3d, yMaxP3d, Pose3d, 1);

    float dxP2d = xMaxP2d - xMinP2d;
    float dyP2d = yMaxP2d - yMinP2d;

    float dxP3d = xMaxP3d - xMinP3d;
    float dyP3d = yMaxP3d - yMinP3d;

    float ratioX = dxP2d / dxP3d;
    float ratioY = dyP2d / dyP3d;

    for (int i=0; i < numJoints; i++) {
        pose3d[i][0] *= ratioX;
        pose3d[i][1] *= ratioY;
        pose3d[i][2] *= ratioX;
    }

    // Shift pelvis to the original point
    vector<float> pelvisPose2d = Pose2d[0];
    vector<float> pelvisPose3d = pose3d[0];
    vector<float> shift3d = {
        -pelvisPose3d[0],
        -pelvisPose3d[1], 
        -pelvisPose3d[2],
    };

    for (int i=0; i < numJoints; i++) {
        for (int j=0; j < dims; j++) {
            pose3d[i][j] += shift3d[j];
        }
    }

    return pose3d;

}

Vector2d ToPixelSpace(Vector2d& PoseIn, int Width, int Height) {

    Vector2d pose = PoseIn;

    int numJoints = pose.size();
    int dims = 3;

    for (int i=0; i < numJoints; i++) {
        pose[i][0] *= Width;
        pose[i][1] *= Height;
        pose[i][2] *= Width;
    }
    
    return pose;

}

void PrintPoint(string Msg, vector<float> &Point, int Dims) {

    string outMsg = "";
    outMsg += Msg;
    outMsg += ": ";
    for (int i=0; i < Dims; i++) {
        outMsg += "[" + std::to_string(Point[i]) + "]";
    }

    cout << outMsg << endl;

}