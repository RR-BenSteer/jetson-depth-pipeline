#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

namespace depthpipe {

// Function to compute scale and shift using least squares
// pair<ArrayXf, ArrayXf> compute_scale_and_shift_ls(const ArrayXXf& prediction, 
//                                                   const ArrayXXf& target, 
//                                                   const ArrayXXf& mask);

class LeastSquaresEstimator {
public:
    LeastSquaresEstimator(const ArrayXXf& estimate, const ArrayXXf& target, const ArrayXXf& valid);

    // Method to compute scale and shift
    void compute_scale_and_shift();

    void apply_scale_and_shift();

    // Clamp output values between specified minimum and maximum values
    // Specify whether depth values are inverted
    void clamp_min_max(float clamp_min = std::numeric_limits<float>::lowest(), 
                       float clamp_max = std::numeric_limits<float>::max(),
                       bool inverted=false);

    // Get the final output
    ArrayXXf get_output() const { return output; }

private:
    ArrayXXf estimate;
    ArrayXXf target;
    ArrayXXf valid;
    float scale, shift;
    ArrayXXf output; // final output 
};

} // namespace depthpipe

#endif // ESTIMATOR_H