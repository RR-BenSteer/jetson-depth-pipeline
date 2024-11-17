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

#include<Eigen/Dense>

#include "cudaSift.h"
#include "cudaImage.h"

#include "DepthAnything.h"
#include "estimator.h"

using namespace std;

namespace depthpipe
{

typedef std::chrono::steady_clock::time_point time_point;

class DepthPipe
{
public:
  DepthPipe(const string &str_setting_path);
  DepthPipe() = delete; // no default constructor

  ~DepthPipe();

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

  void plotPatches(SiftData &siftdata);

  void undistortFisheye(std::vector<cv::Point2d> &points);
  void undistortPerspective(std::vector<cv::Point2d> &points);

  std::vector<double> toQuaternion(const cv::Mat &M);
  Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);

  int n_features;
  float focal_length;
  float baseline;
  float min_depth;
  float max_depth;

  std::shared_ptr<cv::FileStorage> f_settings;

  SiftData siftdata_1;
  SiftData siftdata_2;

  std::shared_ptr<DepthAnything> depthanything_cls;
};

} // namespace DEPTHPIPE

#endif // DEPTHPIPE_H