#include "Infer.h"

void main()
{
	Infer model(true, L"torch_cifar10.onnx", 0);

	cv::Mat img = cv::imread("cifar_ship.PNG", cv::IMREAD_COLOR);

	cv::resize(img, img, cv::Size(32, 32));

	std::vector<float> input_tensor_values = model.Mat2Vec(img, true, true);

	model.PrintInputNode();

	model.SetInputOutputSet();

	model.GetOutput(input_tensor_values, 10, 1);
}