#include <stdio.h>
#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <torch/script.h>


using namespace std;
using namespace cv;

typedef vector<vector<float>> Vector2d;
typedef vector<Vector2d> Vector3d;
typedef vector<Vector3d> Vector4d;


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

/*
input_2D[0, 1, 80]: tensor([[-0.0000,  0.0000],                                                                          
        [-0.0542,  0.0163],                                 
        [-0.0271,  0.2819],                                                                                              
        [-0.0325,  0.5259],                                 
        [ 0.0542, -0.0163],                                                                                              
        [ 0.0651,  0.2169],                                 
        [ 0.2169,  0.2602],
        [-0.0352, -0.1355],
        [-0.0054, -0.2801],
        [ 0.0217, -0.3199],
        [ 0.0258, -0.3741],
        [ 0.0651, -0.2711],
        [ 0.1572, -0.2006],
        [ 0.0651, -0.1030],
        [-0.1193, -0.2711],
        [-0.2060, -0.1681],
        [-0.1084, -0.1030]])
*/


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

void PrintTensorShape(string Msg, at::Tensor& T, int Dims) {

    string outMsg = "";
    outMsg += Msg;
    outMsg += ": ";
    for (int i=0; i < Dims; i++) {
        outMsg += "[" + std::to_string(T.size(i)) + "]";
    }

    cout << outMsg << endl;

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

    torch::Tensor inputTensor = torch::from_blob(InputVec.data(), {batchSize, numFrames, numJoints, dim2d});

    return inputTensor;

}


int main() {

    string msg;
    
    string modelPath = "/home/andrew/projects/MHFormer/checkpoint/pretrained/torchscript_model_traced.pth";
 
    const int batchSize= 1;
    const int numFramesModel= 81;
    const int numJoints = 17;
    const int dim2d = 2;
    const int dim3d = 3;

    Vector2d mockKeypoints = GetMockKeypoints();
    Vector4d mockInputVec = CreateMockInputVec(batchSize, numFramesModel, numJoints, dim2d);

    torch::Tensor inputTensor = CreateInputTensor(mockInputVec);

    //Vector3d inputVec = InitVec3d(numFramesModel, numJoints, dim2d);
    //torch::Tensor inputTensor;
    //inputTensor = torch::from_blob(inputVec.data(), {1, numFramesModel, numJoints, dim2d});


    cout << "Start to load the trained model." << endl;

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(modelPath);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    PrintTensorShape("inputTensor shape", inputTensor, 4);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    at::Tensor outputTensor = model.forward(inputs).toTensor();

    PrintTensorShape("outTensor shape:", outputTensor, 4);

    cout << "End of run." << endl;

    return 0;

}

