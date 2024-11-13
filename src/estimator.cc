#include "estimator.h"

using namespace Eigen;
using namespace std;

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

LeastSquaresEstimator::LeastSquaresEstimator(const ArrayXXf& estimate, const ArrayXXf& target, const ArrayXXf& valid)
    : estimate(estimate), target(target), valid(valid), scale(1.0), shift(0.0) {}

// Method to compute scale and shift
// TODO: Currently only supports batch size of 1
void LeastSquaresEstimator::compute_scale_and_shift() {
    // tie(scale, shift) = compute_scale_and_shift_ls(estimate, target, valid);

    // Sum axes (equivalent to numpy sum over axis 0, 1)
    // ArrayXf a_00 = (valid * estimate * estimate).colwise().sum();
    // ArrayXf a_01 = (valid * estimate).colwise().sum();
    // ArrayXf a_11 = valid.colwise().sum();
    float a_00 = (valid * estimate * estimate).sum();
    float a_01 = (valid * estimate).sum();
    float a_11 = valid.sum();

    float b_0 = (valid * estimate * target).sum();
    float b_1 = (valid * target).sum();
    // cout << "b_0: " << b_0 << endl;
    // cout << "b_1: " << b_1 << endl;

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
}

// Apply scale and shift to the estimate
void LeastSquaresEstimator::apply_scale_and_shift() {
    output = estimate * scale + shift;
}

// Clamp output values between specified minimum and maximum values
void LeastSquaresEstimator::clamp_min_max(float clamp_min, float clamp_max, bool inverted) {
    if (output.size() == 0) {
        cerr << "Output not initialized, call apply_scale_and_shift() first!" << endl;
        return;
    }

    if (clamp_min > 0.0f) {
        if (inverted) {
            float clamp_min_inv = 1.0f / clamp_min;
            output = output.min(clamp_min_inv);
        }
        else
            output = output.max(clamp_min);
    }
    
    if (clamp_max > 0.0f) {
        if (inverted) {
            float clamp_max_inv = 1.0f / clamp_max;
            output = output.max(clamp_max_inv);
        }
        else
            output = output.min(clamp_max);
    }
}

} // namespace depthpipe