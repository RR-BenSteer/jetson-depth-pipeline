#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <iostream>
// #include <Eigen/Dense>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

// using namespace Eigen;
using namespace std;

namespace depthpipe {

class LeastSquaresEstimator {
public:
    // LeastSquaresEstimator(const ArrayXXf& estimate, const ArrayXXf& target, const ArrayXXf& valid);
    LeastSquaresEstimator(const float* estimate, const float* target, const float* valid, int rows, int cols);

    // Method to compute scale and shift
    void compute_scale_and_shift();

    float get_scale() const { return scale; }
    float get_shift() const { return shift; }

    // void apply_scale_and_shift();

    // Clamp output values between specified minimum and maximum values
    // Specify whether depth values are inverted
    void clamp_min_max(cv::cuda::GpuMat &input,
                       float clamp_min = std::numeric_limits<float>::lowest(), 
                       float clamp_max = std::numeric_limits<float>::max(),
                       bool inverted=true);

    // Get the final output
    // ArrayXXf get_output() const { return output; }

private:
    // ArrayXXf estimate;
    // ArrayXXf target;
    // ArrayXXf valid;
    // float scale, shift;
    // ArrayXXf output; // final output 

    const float* estimate;
    const float* target;
    const float* valid;
    int rows;
    int cols;
    float scale;
    float shift;
};

} // namespace depthpipe

#endif // ESTIMATOR_H