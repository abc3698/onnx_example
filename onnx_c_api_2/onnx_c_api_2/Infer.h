#pragma once
#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"
#include "opencv2/opencv.hpp"

class Infer
{
private:	
	std::shared_ptr<Ort::Env> env;
	std::shared_ptr<Ort::Session> session;
	//Ort::Env *env = nullptr;
	//Ort::Session *session = nullptr;
	std::vector<const char*> input_node_names;
	std::vector<int64_t> input_node_dims;
	std::vector<const char*> output_node_names;

public:
	Infer(bool isGPU, const wchar_t * model_path, int deviceID = 0);
	void PrintInputNode();
	void SetInputOutputSet();
	void GetOutput(std::vector<float> &input_tensor, const int clsNum, const int batchNum);
	std::vector<float>  Mat2Vec(cv::Mat &img, bool isColor = true, bool isPytorch = false);
};

