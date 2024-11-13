/**
 * @file
 * @brief The DepthAnything class wraps the DepthAnything network model for monocular depth prediction.
 */

#ifndef DEPTHANYTHING_H
#define DEPTHANYTHING_H

#include <iostream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "model.h"
#include <chrono>

using namespace std;

namespace depthpipe {
	
typedef std::chrono::steady_clock::time_point time_point;

class DepthAnything
{
public:
	DepthAnything(const string &model_path);
	DepthAnything() = delete;
	void compute(const cv::Mat &input, cv::Mat &output);
private:
	depthpipe::Model model;
};

} // namespace depthpipe

#endif // DEPTHANYTHING_H