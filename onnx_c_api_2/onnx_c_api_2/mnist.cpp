#include "Infer.h"
#include "opencv2/opencv.hpp"
void main()
{
	Infer model(true, L"mnist.onnx", 0);

	cv::Mat img = cv::imread("img_2.jpg", cv::IMREAD_GRAYSCALE);		

	cv::resize(img, img, cv::Size(28, 28));	

	std::vector<float> input_tensor_values = model.Mat2Vec(img);
	
	model.PrintInputNode();

	/*model.SetInputOutputSet();	

	model.GetOutput(input_tensor_values, 10);	*/
}