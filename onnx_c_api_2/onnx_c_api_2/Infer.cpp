#include "Infer.h"
#include <assert.h>
#include <iostream>
Infer::Infer(bool isGPU, const wchar_t * model_path, int deviceID)	
{
	env = std::shared_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test"));
	
	Ort::SessionOptions session_options;
	session_options.SetThreadPoolSize(1);

	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

	session_options.SetGraphOptimizationLevel(1);
	
	session = std::shared_ptr<Ort::Session>(new Ort::Session(*env, model_path, session_options));	

	std::cout << "Model Read Success" << std::endl;
}

void Infer::SetInputOutputSet()
{
	Ort::Allocator allocator = Ort::Allocator::CreateDefault();
	input_node_names.reserve(session->GetInputCount());
	input_node_names[0] = session->GetInputName(0, allocator);
	input_node_dims = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

	char* output_name = session->GetOutputName(0, allocator);
	output_node_names.reserve(session->GetOutputCount());
	output_node_names[0] = session->GetOutputName(0, allocator);
}

void Infer::PrintInputNode()
{
	Ort::Allocator allocator = Ort::Allocator::CreateDefault();

	size_t num_input_nodes = session->GetInputCount();
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;  
	std::vector<int64_t> output_node_dims;
										   

	printf("Number of inputs = %zu\n", num_input_nodes);

	// iterate over all input nodes
	for (int i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name = session->GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// print input node types
		Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// print input shapes/dims
		input_node_dims = tensor_info.GetShape();
		printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
		for (int j = 0; j < input_node_dims.size(); j++)
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);

		char* output_name = session->GetOutputName(i, allocator);
		printf("output %d : name=%s\n", i, output_name);

		type_info = session->GetOutputTypeInfo(i);

		output_node_dims = tensor_info.GetShape();
		printf("output %d : num_dims=%zu\n", i, output_node_dims.size());		

		for (int j = 0; j < output_node_dims.size(); j++)
			printf("output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
	}	
}

void Infer::GetOutput(std::vector<float>& input_tensor_values, const int clsNum, const int batchNum)
{	
	// create input tensor object from data values
	Ort::AllocatorInfo allocator_info = Ort::AllocatorInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor(allocator_info, input_tensor_values.data(),
		input_tensor_values.size() * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
	
	// score model & input tensor, get back output tensor
	auto option = Ort::RunOptions(nullptr);
	auto output_tensors = session->Run(option, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);

	// Get pointer to output tensor float values
	float* floatarr = output_tensors.front().GetTensorMutableData<float>();
	
	// score the model, and print scores for first 5 classes	
	
	for (int batch = 0; batch < batchNum; ++batch)
	{
		double max = -DBL_MAX;
		std::cout << max << std::endl;
		int idx;
		for (int i = 0; i < clsNum; ++i)
		{									
			if (floatarr[i + (batch * clsNum)] > max)
			{
				max = floatarr[i + (batch * clsNum)];
				idx = i;
			}
		}
		printf("Batch [%d] Score for class [%d] =  %f\n", batch, idx, max);
		std::cout << std::endl;
	}	
}

std::vector<float> Infer::Mat2Vec(cv::Mat & img, bool isColor, bool isPytorch)
{
	cv::Mat img_float;
	int channel = (isColor) ? 3 : 1;
	if (isColor)
	{
		img.convertTo(img_float, CV_32FC3);
	}
	else
	{
		img.convertTo(img_float, CV_32FC1);
	}
	
	img_float /= 255.;

	if(isPytorch && isColor)
		img_float.reshape(channel, img_float.cols * img_float.rows);

	std::vector<float> input_tensor_values;
	if (img_float.isContinuous()) {
		input_tensor_values.assign((float*)img_float.data, (float*)img_float.data + img_float.total() * channel);
	}
	else {
		for (int i = 0; i < img_float.rows; ++i) {
			input_tensor_values.insert(input_tensor_values.end(), img_float.ptr<float>(i), img_float.ptr<float>(i) + img_float.cols * channel);
		}
	}

	return input_tensor_values;
}
