#include "DepthAnything.h"

using namespace std;

namespace depthpipe
{
	DepthAnything::DepthAnything(const string &model_path) : model(model_path) {}

	void DepthAnything::compute(const cv::Mat &input, cv::Mat &output) {
		model.run(input, output);
	}

} // namespace depthpipe