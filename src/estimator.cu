#include "estimator.h"

// using namespace Eigen;
using namespace std;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// CUDA Kernel for element-wise operations
__global__ void compute_products(const float* estimate, const float* target, const float* valid, 
                                  float* valid_estimate_estimate_sum, float* valid_estimate_sum, float* valid_sum,
                                  float* valid_estimate_target_sum, float* valid_target_sum, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        float val = valid[idx];
        float est = estimate[idx];
        float tgt = target[idx];

        atomicAdd(valid_estimate_estimate_sum, val * est * est);
        atomicAdd(valid_estimate_sum, val * est);
        atomicAdd(valid_sum, val);
        atomicAdd(valid_estimate_target_sum, val * est * tgt);
        atomicAdd(valid_target_sum, val * tgt);
    }
}

__global__ void clampKernel(float* data, int rows, int cols, size_t pitch, float minVal, float maxVal) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        float* row = (float*)((char*)data + y * pitch);
        row[x] = fminf(fmaxf(row[x], minVal), maxVal);
    }
}

namespace depthpipe {

// // Function to compute scale and shift using least squares
// pair<ArrayXf, ArrayXf> compute_scale_and_shift_ls(const ArrayXXf& prediction, 
//                                                   const ArrayXXf& target, 
//                                                   const ArrayXXf& mask) {
//     // Sum axes (equivalent to numpy sum over axis 0, 1)
//     ArrayXf a_00 = (mask * prediction * prediction).colwise().sum();
//     ArrayXf a_01 = (mask * prediction).colwise().sum();
//     ArrayXf a_11 = mask.colwise().sum();

//     ArrayXf b_0 = (mask * prediction * target).colwise().sum();
//     ArrayXf b_1 = (mask * target).colwise().sum();

//     ArrayXf x_0 = ArrayXf::Zero(b_0.size());
//     ArrayXf x_1 = ArrayXf::Zero(b_1.size());

//     // Calculate determinant
//     ArrayXf det = a_00 * a_11 - a_01 * a_01;
//     Array<bool, Dynamic, 1> valid = det > 0;

//     // Scale and shift computation
//     x_0 = (a_11 * b_0 - a_01 * b_1).array() / det;
//     x_1 = (-a_01 * b_0 + a_00 * b_1).array() / det;

//     // Apply validity mask
//     x_0 = valid.select(x_0, 0);
//     x_1 = valid.select(x_1, 0);

//     return {x_0, x_1};
// }

// LeastSquaresEstimator::LeastSquaresEstimator(const ArrayXXf& estimate, const ArrayXXf& target, const ArrayXXf& valid)
    // : estimate(estimate), target(target), valid(valid), scale(1.0), shift(0.0) {}
LeastSquaresEstimator::LeastSquaresEstimator(const float* estimate, const float* target, const float* valid, int rows, int cols)
    : estimate(estimate), target(target), valid(valid), rows(rows), cols(cols), scale(1.0), shift(0.0) {}


// Method to compute scale and shift
// TODO: Currently only supports batch size of 1
// void LeastSquaresEstimator::compute_scale_and_shift() {
//     // tie(scale, shift) = compute_scale_and_shift_ls(estimate, target, valid);

//     // Sum axes (equivalent to numpy sum over axis 0, 1)
//     // ArrayXf a_00 = (valid * estimate * estimate).colwise().sum();
//     // ArrayXf a_01 = (valid * estimate).colwise().sum();
//     // ArrayXf a_11 = valid.colwise().sum();
//     float a_00 = (valid * estimate * estimate).sum();
//     float a_01 = (valid * estimate).sum();
//     float a_11 = valid.sum();

//     float b_0 = (valid * estimate * target).sum();
//     float b_1 = (valid * target).sum();
//     // cout << "b_0: " << b_0 << endl;
//     // cout << "b_1: " << b_1 << endl;

//     float x_0 = 0.0;
//     float x_1 = 0.0;

//     // Calculate determinant
//     float det = a_00 * a_11 - a_01 * a_01;
//     // Array<bool, Dynamic, 1> mask = det > 0;
//     if (det <= 0) {
//         cout << "Matrix not positive definite. Not scaling depth." << endl;
//         scale = 1.0;
//         shift = 0.0;
//         return;
//     }

//     // Scale and shift computation
//     x_0 = (a_11 * b_0 - a_01 * b_1) / det;
//     x_1 = (-a_01 * b_0 + a_00 * b_1) / det;

//     scale = x_0;
//     shift = x_1;
// }

// Compute scale and shift using CUDA
void LeastSquaresEstimator::compute_scale_and_shift() {
    // Allocate GPU memory
    float *d_estimate, *d_target, *d_valid;
    float *d_valid_estimate_estimate_sum, *d_valid_estimate_sum, *d_valid_sum, *d_valid_target_sum, *d_valid_estimate_target_sum;
    CUDA_CHECK(cudaMalloc(&d_estimate, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_target, rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid, rows * cols * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_valid_estimate_estimate_sum,  sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid_estimate_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid_target_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_valid_estimate_target_sum, sizeof(float)));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_estimate, estimate, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, target, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_valid, valid, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize output arrays on GPU
    CUDA_CHECK(cudaMemset(d_valid_estimate_estimate_sum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_valid_estimate_sum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_valid_sum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_valid_target_sum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_valid_estimate_target_sum, 0, sizeof(float)));

    // Launch kernel
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / threads;
    compute_products<<<blocks, threads>>>(d_estimate, d_target, d_valid, 
                                          d_valid_estimate_estimate_sum, d_valid_estimate_sum, d_valid_sum,
                                          d_valid_estimate_target_sum, d_valid_target_sum, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to CPU
    float a_00, a_01, a_11, b_0, b_1;
    CUDA_CHECK(cudaMemcpy(&a_00, d_valid_estimate_estimate_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&a_01, d_valid_estimate_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&a_11, d_valid_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&b_0, d_valid_estimate_target_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&b_1, d_valid_target_sum, sizeof(float), cudaMemcpyDeviceToHost));

//     float a_00 = (valid * estimate * estimate).sum();
//     float a_01 = (valid * estimate).sum();
//     float a_11 = valid.sum();

//     float b_0 = (valid * estimate * target).sum();
//     float b_1 = (valid * target).sum();

    // Perform final reduction on CPU
    // float a_00 = 0, a_01 = 0, a_11 = 0, b_0 = 0, b_1 = 0;

    // Compute determinant and scale/shift
    // float det = a_00 * a_11 - a_01 * a_01;
    // if (det <= 0) {
    //     std::cerr << "Matrix not positive definite. Not scaling depth." << std::endl;
    //     scale = 1.0;
    //     shift = 0.0;
    // } else {
    //     scale = (a_11 * b_0 - a_01 * b_1) / det;
    //     shift = (-a_01 * b_0 + a_00 * b_1) / det;
    // }

    float x_0 = 0.0;
    float x_1 = 0.0;

    // Calculate determinant
    float det = a_00 * a_11 - a_01 * a_01;
    // Array<bool, Dynamic, 1> mask = det > 0;
    if (det <= 0) {
        cout << "Matrix not positive definite. Not scaling depth." << endl;
        scale = 1.0;
        shift = 0.0;
        return;
    }

    // Scale and shift computation
    x_0 = (a_11 * b_0 - a_01 * b_1) / det;
    x_1 = (-a_01 * b_0 + a_00 * b_1) / det;

    scale = x_0;
    shift = x_1;

    // Free GPU memory
    cudaFree(d_estimate);
    cudaFree(d_target);
    cudaFree(d_valid);
    cudaFree(d_valid_estimate_estimate_sum);
    cudaFree(d_valid_estimate_sum);
    cudaFree(d_valid_sum);
    cudaFree(d_valid_target_sum);
    cudaFree(d_valid_estimate_target_sum);
}

// Apply scale and shift to the estimate
// void LeastSquaresEstimator::apply_scale_and_shift() {
    // output = estimate * scale + shift;
    // output.create(relDepthMap.size(), relDepthMap.type());
// }


// Clamp output values between specified minimum and maximum values
void LeastSquaresEstimator::clamp_min_max(cv::cuda::GpuMat &input, float clamp_min, float clamp_max, bool inverted)
{
    // Launch kernel
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    if (inverted)
        clampKernel<<<blocks, threads>>>(input.ptr<float>(), rows, cols, input.step, 1.0f/clamp_max, 1.0f/clamp_min);
    else
        clampKernel<<<blocks, threads>>>(input.ptr<float>(), rows, cols, input.step, clamp_min, clamp_max);
}

// Clamp output values between specified minimum and maximum values
// void LeastSquaresEstimator::clamp_min_max(float clamp_min, float clamp_max, bool inverted) {
    // if (output.size() == 0) {
        // cerr << "Output not initialized, call apply_scale_and_shift() first!" << endl;
        // return;
    // }

//     if (clamp_min > 0.0f) {
//         if (inverted) {
//             float clamp_min_inv = 1.0f / clamp_min;
//             output = output.min(clamp_min_inv);
//         }
//         else
//             output = output.max(clamp_min);
//     }
    
//     if (clamp_max > 0.0f) {
//         if (inverted) {
//             float clamp_max_inv = 1.0f / clamp_max;
//             output = output.max(clamp_max_inv);
//         }
//         else
//             output = output.min(clamp_max);
//     }
// }

} // namespace depthpipe