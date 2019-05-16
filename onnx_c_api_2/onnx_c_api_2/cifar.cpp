#include "Infer.h"

void main()
{
	Infer model(true, L"cifar10.onnx", 0);

	cv::Mat img = cv::imread("cifar_ship.png", cv::IMREAD_COLOR);	
	cv::resize(img, img, cv::Size(32, 32));	

	std::vector<float> input_tensor_values = model.Mat2Vec(img);	
	
	cv::Mat img2 = cv::imread("cifar_cat.png", cv::IMREAD_COLOR);
	cv::resize(img2, img2, cv::Size(32, 32));

	std::vector<float> input_tensor_values2 = model.Mat2Vec(img2);
	input_tensor_values.insert(input_tensor_values.end(), 
		input_tensor_values2.begin(), input_tensor_values2.end());

	model.PrintInputNode();

	model.SetInputOutputSet();
	
	model.GetOutput(input_tensor_values, 10, 2);
}