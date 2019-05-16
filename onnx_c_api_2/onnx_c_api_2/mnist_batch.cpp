#include "Infer.h"

void main()
{
	Infer model(true, L"mnist.onnx", 0);

	cv::Mat img = cv::imread("img_1.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread("img_2.jpg", cv::IMREAD_GRAYSCALE);

	cv::resize(img, img, cv::Size(28, 28));
	cv::resize(img2, img2, cv::Size(28, 28));

	std::vector<float> input_tensor_values = model.Mat2Vec(img);
	std::vector<float> input_tensor_values2 = model.Mat2Vec(img2);

	input_tensor_values.insert(input_tensor_values.end(), input_tensor_values2.begin(), input_tensor_values2.end());
	std::cout << input_tensor_values .size() << std::endl;
	
	model.PrintInputNode();

	model.SetInputOutputSet();

	model.GetOutput(input_tensor_values, 10, 2);		
}