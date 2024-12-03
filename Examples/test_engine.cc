#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace nvinfer1;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Filter out INFO and lower severity messages for cleaner output
        if (severity <= Severity::kINFO) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Assume you already have a serialized TensorRT engine file.
    const char* engineFile = "depth_anything_v2_vits.engine.Xavier.fp32.4.1";
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open engine file!" << std::endl;
        return -1;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    file.close();

    Logger gLogger;

    // Create runtime and deserialize engine
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fileSize);
    if (!engine) {
        std::cerr << "Failed to create CUDA engine!" << std::endl;
        return -1;
    }

    // Create execution context
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context!" << std::endl;
        return -1;
    }

    // Allocate device memory for input and output
    const int inputIndex = engine->getBindingIndex("input"); // Replace "input" with your input tensor name
    const int outputIndex = engine->getBindingIndex("output"); // Replace "output" with your output tensor name

    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    std::cout << "Input dimensions: ";
    for (int i = 0; i < inputDims.nbDims; i++) {
        std::cout << inputDims.d[i] << " ";
    }
    std::cout << std::endl;

    inputDims.d[0] = 1;  // Batch size
    inputDims.d[1] = 3;  // Channels
    inputDims.d[2] = 518; // Height
    inputDims.d[3] = 518; // Width
    if (!context->setBindingDimensions(inputIndex, inputDims)) {
        std::cerr << "Failed to set binding dimensions!" << std::endl;
        return -1;
    }

    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "Not all input dimensions are specified!" << std::endl;
        return -1;
    }

    size_t inputSize = 1 * 3 * 518 * 518 * sizeof(float);
    size_t outputSize = 1 * 518 * 518 * sizeof(float);

    void* dInput = nullptr;
    void* dOutput = nullptr;
    checkCudaError(cudaMalloc(&dInput, inputSize), "Failed to allocate device memory for input");
    checkCudaError(cudaMalloc(&dOutput, outputSize), "Failed to allocate device memory for output");

    void* bindings[2] = {dInput, dOutput};

    // Create a CUDA stream
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create CUDA stream");

    // Prepare input data (e.g., copy from host to device)
    std::vector<float> hInput(3 * 518 * 518, 1.0f); // Example input data
    checkCudaError(cudaMemcpyAsync(dInput, hInput.data(), inputSize, cudaMemcpyHostToDevice, stream),
                   "Failed to copy input data to device");

    // Run inference using enqueueV3
    if (!context->enqueueV2(bindings, stream, nullptr)) {
        std::cerr << "Failed to execute inference with enqueueV2!" << std::endl;
        return -1;
    }

    // Copy output data from device to host
    std::vector<float> hOutput(518*518); // Example output buffer
    checkCudaError(cudaMemcpyAsync(hOutput.data(), dOutput, outputSize, cudaMemcpyDeviceToHost, stream),
                   "Failed to copy output data to host");

    // Wait for stream to complete
    checkCudaError(cudaStreamSynchronize(stream), "Failed to synchronize CUDA stream");

    // Process output (example: print the first few results)
    std::cout << "Output: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << hOutput[i] << " ";
    }
    std::cout << std::endl;

    // Clean up
    cudaStreamDestroy(stream);
    cudaFree(dInput);
    cudaFree(dOutput);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
