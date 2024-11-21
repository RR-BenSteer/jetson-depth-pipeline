#include "model.h"

using namespace std;

namespace depthpipe {

Model::Model(const string &engine_path) {
    // Specify our GPU inference configuration options
    // Specify what precision to use for inference
    // FP16 is approximately twice as fast as FP32.
    _options.precision = Precision::FP16;
    // If using INT8 precision, must specify path to directory containing
    // calibration data.
    _options.calibrationDataDirectoryPath = "";
    // Specify the batch size to optimize for.
    _options.optBatchSize = 1;
    // Specify the maximum batch size we plan on running.
    _options.maxBatchSize = 4;

    // Set Engine pointer equal to new engine class
    _engine = std::make_unique<Engine<float>>(_options);

    // Normalize values between
    // [0.f, 1.f] so we use the following params
    // std::array<float, 3> subVals{0.f, 0.f, 0.f};
    // std::array<float, 3> divVals{1.f, 1.f, 1.f};
    std::array<float, 3> subVals{0.485f, 0.456f, 0.406f};
    std::array<float, 3> divVals{0.229f, 0.224f, 0.225f};
    bool normalize = true;
    // If the model requires values to be normalized between [-1.f, 1.f], use the
    // following params:
    //    subVals = {0.5f, 0.5f, 0.5f};
    //    divVals = {0.5f, 0.5f, 0.5f};
    //    normalize = true;

    // note should be onnx model path
    if (!engine_path.empty()) {
        // Build the onnx model into a TensorRT engine file, and load the TensorRT
        // engine file into memory.
        bool succ = _engine->buildLoadNetwork(engine_path, subVals, divVals, normalize);
        if (!succ) {
            throw std::runtime_error("Unable to build or load TensorRT engine.");
        }
    } else {
        throw std::runtime_error("TensorRT engine path is empty.");
    }
}

Model::~Model() {
}

int32_t Model::getBatchSize() const noexcept {
    return _options.optBatchSize;
}

void Model::run(const cv::Mat &cpuImg, cv::Mat &depth){
    // Upload the image GPU memory
    cv::cuda::GpuMat img;
    img.upload(cpuImg);

    // The model expects RGB input
    cv::cuda::cvtColor(img, img, cv::COLOR_BGR2RGB);

    const auto inputDims = _engine->getInputDims();
    const auto inputDim = inputDims[0];
    std::vector<std::vector<cv::cuda::GpuMat>> inputs;

    std::vector<cv::cuda::GpuMat> input1;
    // You can choose to resize by scaling, adding padding, or a combination
    // of the two in order to maintain the aspect ratio You can use the
    // Engine::resizeKeepAspectRatioPadRightBottom to resize to a square while
    // maintain the aspect ratio (adds padding where necessary to achieve
    // this).
    int unpad_width, unpad_height;
    auto resized = Engine<float>::resizeKeepAspectRatioPadRightBottom(img, inputDim.d[1], inputDim.d[2], unpad_height, unpad_width, 14);
    // You could also perform a resize operation without maintaining aspect
    // ratio with the use of padding by using the following instead:
    //            cv::cuda::resize(img, resized, cv::Size(inputDim.d[2],
    //            inputDim.d[1])); // TRT dims are (height, width) whereas
    //            OpenCV is (width, height)

    input1.emplace_back(std::move(resized));
    inputs.emplace_back(std::move(input1));

    bool succ = _engine->runInference(inputs, depth);
    if (!succ) {
        throw std::runtime_error("Unable to run inference.");
    }

    // Rescale back to original image size
    Engine<float>::extractAndResizeROI(depth, depth, unpad_height, unpad_width, img.rows, img.cols);
}

} // namespace depthpipe
