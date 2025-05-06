/**
 * @file
 * @brief The base class for wrapping a network model interface to the tensorrt-cpp-api.
 */

// #pragma once

#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>

#include "engine.h" // tensorrt-cpp-api

using namespace std;

namespace depthpipe {

class Model
{
public:
	// Create trt engine from engine path
	Model(const string &engine_path);

	Model() = delete;

	~Model();

	void run(const cv::Mat &cpuImg, cv::Mat &depth);

	int32_t getBatchSize() const noexcept;

private:
    std::unique_ptr<Engine<float>> _engine;
    Options _options;
};

} // namespace depthpipe

#endif // MODEL_H