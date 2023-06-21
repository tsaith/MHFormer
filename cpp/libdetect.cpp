#include "libdetect.h"

Vector2d GetMockKeypoints() {

    Vector2d keypoints {

        { 0.0000,  0.0000},
        {-0.0542, -0.0163},
        {-0.0651,  0.2169},
        {-0.2169,  0.2602},
        { 0.0542,  0.0163},
        { 0.0271,  0.2819},
        { 0.0325,  0.5259},
        { 0.0352, -0.1355},
        { 0.0054, -0.2801},
        {-0.0217, -0.3199},
        {-0.0258, -0.3741},
        { 0.1193, -0.2711},
        { 0.2060, -0.1681},
        { 0.1084, -0.1030},
        {-0.0651, -0.2711},
        {-0.1572, -0.2006},
        {-0.0651, -0.1030}

    };

    return keypoints;

}

Vector2d GetMockKeypointsFlip() {

    Vector2d keypoints {
    
        {-0.0000,  0.0000}, 
        {-0.0542,  0.0163},                                                                                              
        {-0.0271,  0.2819},
        {-0.0325,  0.5259},
        { 0.0542, -0.0163},
        { 0.0651,  0.2169},
        { 0.2169,  0.2602},
        {-0.0352, -0.1355},
        {-0.0054, -0.2801},
        { 0.0217, -0.3199},
        { 0.0258, -0.3741},
        { 0.0651, -0.2711},
        { 0.1572, -0.2006},
        { 0.0651, -0.1030},
        {-0.1193, -0.2711},
        {-0.2060, -0.1681},
        {-0.1084, -0.1030}

    };

    return keypoints;

}

Vector2d GetMockOutputs() {
    
    Vector2d outputs {

        {-2.6163e-04,  3.5916e-06,  2.4172e-04},
        {-9.6760e-02, -2.6799e-02,  7.6925e-02},
        {-1.3421e-01,  3.2777e-01,  1.9386e-01},
        { 9.6760e-02,  2.6845e-02, -7.6880e-02},
        { 2.5307e-02,  4.5648e-01, -9.2942e-02},
        { 1.2370e-01,  8.6109e-01,  5.3697e-02},
        { 5.4214e-02, -2.4407e-01, -5.9054e-02},
        { 3.0803e-02, -4.7703e-01, -1.6575e-01},
        {-2.0972e-02, -5.6478e-01, -2.3580e-01},
        {-3.8528e-02, -6.6981e-01, -1.9677e-01},
        { 1.9380e-01, -4.2013e-01, -2.1035e-01},
        { 4.1497e-01, -2.3785e-01, -2.4411e-01},
        { 2.1095e-01, -2.3227e-01, -3.1787e-01},
        {-1.2163e-01, -4.6578e-01, -6.3605e-02},
        {-3.2557e-01, -3.2365e-01, -2.3023e-02},
        {-2.8519e-01, -1.4468e-01, -1.6588e-01}

    };

    return outputs;

}

Vector2d GetMockOutputsFlip() {
    
    Vector2d outputs {

        {-1.4440e-04,  2.4511e-04,  1.1604e-04},
        {-9.9115e-02,  3.7352e-02, -7.5168e-02},
        { 8.5948e-03,  4.3601e-01, -1.4788e-01},
        {-9.8936e-02,  8.5636e-01, -6.3626e-02},
        { 9.9062e-02, -3.7373e-02,  7.5215e-02},
        { 1.6354e-01,  3.3519e-01,  1.1755e-01},
        { 1.9216e-01,  5.4643e-01,  4.8009e-01},
        {-4.4273e-02, -2.4942e-01, -6.7804e-02},
        {-1.3177e-02, -4.7792e-01, -1.8374e-01},
        { 3.4882e-02, -5.5918e-01, -2.5274e-01},
        { 4.6720e-02, -6.6495e-01, -2.2541e-01},
        { 1.3001e-01, -4.6374e-01, -7.0558e-02},
        { 3.4332e-01, -3.5235e-01,  1.0831e-01},
        { 3.1033e-01, -1.4761e-01,  3.5555e-02},
        {-1.8859e-01, -4.1598e-01, -2.2971e-01},
        {-3.9814e-01, -2.3935e-01, -2.4810e-01},
        {-1.8398e-01, -2.4943e-01, -3.3361e-01}

    };

    return outputs;

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

Vector4d CreateMockInputVec(int BatchSize, int NumFrames, int NumJoints, int Dim2d) {

    Vector2d keypoints = GetMockKeypoints();
    Vector4d inputVec = InitVec4d(BatchSize, NumFrames, NumJoints, Dim2d);

    for (int ib=0; ib < BatchSize; ib++) {
        for (int i=0; i < NumFrames; i++) {
            for (int j=0; j < NumJoints; j++) {
                for (int k=0; k < Dim2d; k++) {
                    inputVec[ib][i][j][k] = keypoints[j][k];
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


    //torch::Tensor inputTensor = torch::from_blob(InputVec.data(), {batchSize, numFrames, numJoints, dim2d});
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

Vector2d GetPoseOut(at::Tensor& Outputs) {

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

    // Shift the pelvis
    vector<float> pelvis = pose3d[0];
    for (int i=0; i < numJoints; i++) {
        for (int j=0; j < dims; j++) {
            pose3d[i][j] -= pelvis[j];
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