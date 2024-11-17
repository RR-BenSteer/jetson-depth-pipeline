#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <string>
#include <random>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "DepthPipe.h"

namespace fs = std::filesystem;
using namespace std;
using namespace depthpipe;

void fillSparseDepthMap(const cv::Mat gt_depth, cv::Mat &depth_map, cv::Mat &depth_mask, const int num_pts, const float max_depth, bool inverted);
double computeDepthRMSE(const cv::Mat& depthPred, const cv::Mat& depthGT, double minDepth, double maxDepth);

int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./stereo_test settings_file path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> left_image_paths;
    vector<string> right_image_paths;
    vector<string> depth_image_paths;

    string base_dir = string(argv[2]);
    string image_path = base_dir + "/16233051861828957.tiff";
    string depth_path = base_dir + "/16233051861828957_SeaErra_abs_depth.tif";

    cv::Mat im = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    cv::Mat gt_depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);


    cv::Mat sparse_depth, sparse_mask;
    fillSparseDepthMap(gt_depth, sparse_depth, sparse_mask, 500, 5.0, true);

    DepthPipe depthpipe(argv[1]);

    cv::Mat depth_map;
    depthpipe.process(im, sparse_depth, sparse_mask, depth_map);

    double rmse = computeDepthRMSE(depth_map, gt_depth, 0.1, 2.0);
    cout << "RMSE: " << rmse << endl;

    // plot images
    // double minVal, maxVal;
    // cv::minMaxLoc(depth_map, &minVal, &maxVal);
    // cout << "Max pred depth: " << maxVal << ", Min pred depth: " << minVal << endl;
    cv::Mat depthViz = depth_map.clone() * 255.0 / 10.0;
    depthViz.convertTo(depthViz, CV_8U);
    cv::Mat depthColor;
    cv::applyColorMap(depthViz, depthColor, cv::COLORMAP_JET);
    cv::namedWindow("Pred Depth", cv::WINDOW_NORMAL);
    cv::imshow("Pred Depth", depthColor);

    // cv::minMaxLoc(gt_depth, &minVal, &maxVal);
    // cout << "Max gt depth: " << maxVal << ", Min gt depth: " << minVal << endl;
    cv::Mat depthViz2 = gt_depth.clone() * 255.0 / 10.0;
    depthViz2.convertTo(depthViz2, CV_8U);
    cv::Mat depthColor2;
    cv::applyColorMap(depthViz2, depthColor2, cv::COLORMAP_JET);
    cv::namedWindow("GT Depth", cv::WINDOW_NORMAL);
    cv::imshow("GT Depth", depthColor2);

    cv::namedWindow("Image", cv::WINDOW_NORMAL);
    cv::imshow("Image", im);
    cv::waitKey(0);

    return 0;
}

double computeDepthRMSE(const cv::Mat& depthPred, const cv::Mat& depthGT, double minDepth, double maxDepth) {
    // Ensure the input depth masks have the same size and type
    if (depthPred.size() != depthGT.size() || depthPred.type() != depthGT.type()) {
        std::cerr << "Error: Depth masks must have the same size and type." << std::endl;
        return -1.0;
    }

    // Create a mask based on the valid depth range in the ground truth
    cv::Mat validMask = (depthGT >= minDepth) & (depthGT <= maxDepth);

    // Mask out the invalid depth values in the ground truth
    cv::Mat maskedDepthGT, maskedDepthPred;
    depthGT.copyTo(maskedDepthGT, validMask);
    depthPred.copyTo(maskedDepthPred, validMask);

    // Calculate the squared differences only for the valid pixels
    cv::Mat diff;
    cv::absdiff(maskedDepthGT, maskedDepthPred, diff);
    diff = diff.mul(diff); // Square the differences

    // Compute the mean of the squared differences (ignoring masked-out values)
    double mse = cv::mean(diff, validMask)[0];

    // Compute the Root Mean Squared Error (RMSE)
    return std::sqrt(mse);
}

  void fillSparseDepthMap(const cv::Mat gt_depth, cv::Mat &depth_map, cv::Mat &depth_mask, const int num_pts, const float max_depth, bool inverted)
  {
    // Input depth map should be correct size
    depth_map = cv::Mat::zeros(gt_depth.rows, gt_depth.cols, CV_32F);
    depth_mask = cv::Mat::zeros(gt_depth.rows, gt_depth.cols, CV_32F);

    // Create a random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distRows(0, gt_depth.rows - 1);
    std::uniform_int_distribution<int> distCols(0, gt_depth.cols - 1);

    for (int i = 0; i < num_pts; i++) {
      int row = distRows(rng);
      int col = distCols(rng);
      if (gt_depth.at<float>(row, col) <= 0.0 || gt_depth.at<float>(row, col) > max_depth) {
        i--;
        continue;
      }
      if (inverted)
        depth_map.at<float>(row, col) = 1.0 / gt_depth.at<float>(row, col);
      else
        depth_map.at<float>(row, col) = gt_depth.at<float>(row, col);
      depth_mask.at<float>(row, col) = 1.0;
    }
  }