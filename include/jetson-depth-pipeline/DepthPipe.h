/**
 * @file
 * @brief The DepthPipe class is an implementation of a real-time depth estimation pipeline on Jetson devices for stereo images.
 */

#ifndef DEPTHPIPE_H
#define DEPTHPIPE_H

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Dense>

#include "cudaSift.h"
#include "cudaImage.h"

#include "jetson-depth-pipeline/DepthAnything.h"
#include "jetson-depth-pipeline/estimator.h"

using namespace std;

namespace depthpipe
{

struct DepthPipeSettings {
  int n_features;
  int sift_max_dim;
  float min_depth;
  float max_depth;
  float match_disp_tolerance;
  std::string model_path;
};

typedef std::chrono::steady_clock::time_point time_point;

class DepthPipe
{
public:
  DepthPipe(const DepthPipeSettings &settings);
  DepthPipe(const string &str_setting_path);
  DepthPipe() = delete; // no default constructor

  ~DepthPipe();

  void init (const DepthPipeSettings &settings);
  void setCameraProperties (float focalLength, float baseline, float cx, float cy);

  float process(const cv::Mat &im1, const cv::Mat &im2, cv::Mat &depthMap);
  void process(const cv::Mat &im, cv::Mat &sparseDepthMap, cv::Mat &sparseDepthMask, cv::Mat &depthMap);

private:
  void computeFeaturesCUDA(const cv::Mat& I1, const cv::Mat& I2,
    std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc1, std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc2);

  void matchCUDA(std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc1,
    std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc2, std::vector<cv::DMatch> &matches,
    const float ratio_thresh, const bool cross_check=false);

  std::vector<double> findInliers(const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, 
    const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inliers, bool is_fish1, bool is_fish2);
  
  void CUDAtoCV(SiftData &siftdata1, SiftData &siftdata2, std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc1,
    std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc2);

  int fillSparseDepthMap(const std::vector<cv::KeyPoint> &kpts_left, const std::vector<cv::KeyPoint> &kpts_right,
    const std::vector<cv::DMatch> &matches, cv::Mat &depth_map, cv::Mat &depth_mask, bool inverted=true);

  void computeMetricGlobalScaleAndShift(cv::Mat &relDepthMap, cv::Mat &depthMap,
    std::vector<cv::KeyPoint> &kpts1, std::vector<cv::KeyPoint> &kpts2, std::vector<cv::DMatch> &inliers);

  void plotPatches(SiftData &siftdata);

  void undistortFisheye(std::vector<cv::Point2d> &points);
  void undistortPerspective(std::vector<cv::Point2d> &points);

  std::vector<double> toQuaternion(const cv::Mat &M);
  Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

  // pipeline settings
  int n_features;
  int sift_max_dim;
  float min_depth;
  float max_depth;
  float match_disp_tolerance;

  // camera parameters
  float focal_length;
  cv::Point2d pp;
  float baseline;

  SiftData siftdata_1;
  SiftData siftdata_2;

  std::shared_ptr<DepthAnything> depthanything_cls;
};

} // namespace DEPTHPIPE

#endif // DEPTHPIPE_H