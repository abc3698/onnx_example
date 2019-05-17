# onnx_example
Visual Studio에서 사용할 수 있는 예제 코드

# 준비 조건
- Visual studio 2017 이상
- Python 3.6 버전
- Opencv (프로젝트에서 4.1 버젼 사용)
- CUDA 10.0 Cudnn 7 버전
- onnx 배치 크기를 바꾸기 위해서 WinMLDashboard 사용(https://github.com/Microsoft/Windows-Machine-Learning/releases/tag/v0.6.1)

# 주의 사항
- onnx runtime library에 경우 소스에서 직접 빌드하여 사용
- 디바이스 및 옵션에 맞게 소스에서 빌드하여 사용하기 바람
- Windows Onnx 소스 주소 : https://github.com/microsoft/onnxruntime

# 사용법
- ipnyp를 이용하여 mnist 혹은 cifar10 onnx 생성
- 프로젝트에 onnx 파일을 저장하고 해당하는 코드에 맞게 cpp 파일 실행

# to-do
- [x] Cpp 예제 작성
- [x] Kera to Onnx
- [ ] Tensorflw to Onnx
- [x] Pytorch to Onnx
