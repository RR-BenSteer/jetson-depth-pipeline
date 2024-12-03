#include "DepthPipe.h"

using namespace std;

namespace depthpipe
{

  DepthPipe::DepthPipe(const string &str_setting_path) {
      f_settings.reset(new cv::FileStorage(str_setting_path.c_str(), cv::FileStorage::READ));
      focal_length  = (*f_settings)["Stereo.f"].real();
      baseline      = (*f_settings)["Stereo.baseline"].real();
      n_features    = (*f_settings)["Extractor.nFeatures"].operator int();
      sift_max_dim  = (*f_settings)["Extractor.maxDim"].operator int();
      min_depth     = (*f_settings)["Depth.min"].real();
      max_depth     = (*f_settings)["Depth.max"].real();

      // CudaSift init
      // InitCuda(0);
      InitSiftData(siftdata_1, n_features, true, true);
      InitSiftData(siftdata_2, n_features, true, true);

      // DepthAnything init
      depthanything_cls = shared_ptr<DepthAnything>(new DepthAnything((*f_settings)["Depth.model"]));
  }

  DepthPipe::~DepthPipe() {
    FreeSiftData(siftdata_1);
    FreeSiftData(siftdata_2);
  }

  void DepthPipe::process(const cv::Mat &im, cv::Mat &sparseDepthMap, cv::Mat &sparseDepthMask, cv::Mat &depthMap)
  {
    cv::Mat relDepthMap;
    depthanything_cls->compute(im, relDepthMap);

    // compute metric depth by scale and shift
    ArrayXXf estimate = Map<ArrayXXf>(relDepthMap.ptr<float>(), relDepthMap.rows, relDepthMap.cols);
    ArrayXXf target   = Map<ArrayXXf>(sparseDepthMap.ptr<float>(), sparseDepthMap.rows, sparseDepthMap.cols);
    ArrayXXf valid    = Map<ArrayXXf>(sparseDepthMask.ptr<float>(), sparseDepthMask.rows, sparseDepthMask.cols);

    LeastSquaresEstimator estimator(estimate, target, valid);
    estimator.compute_scale_and_shift();
    estimator.apply_scale_and_shift();
    estimator.clamp_min_max(min_depth, max_depth, true);

    ArrayXXf output = estimator.get_output();
    output = output.inverse(); // invert to metric depth
    depthMap = cv::Mat(output.rows(), output.cols(), CV_32F, const_cast<float*>(output.data())).clone();
  }

  float DepthPipe::process(const cv::Mat &im1, const cv::Mat &im2, cv::Mat &depthMap)
  {
    time_point t1 = std::chrono::steady_clock::now();

  	// compute features
  	std::vector<cv::KeyPoint> kpts1, kpts2;
  	cv::Mat desc1, desc2;
  	computeFeaturesCUDA(im1, im2, kpts1, desc1, kpts2, desc2);
    time_point t2 = std::chrono::steady_clock::now();

  	// match features
  	std::vector<cv::DMatch> matches;
  	matchCUDA(kpts1, desc1, kpts2, desc2, matches, 0.95, true);
    time_point t3 = std::chrono::steady_clock::now();

  	// find inliers
  	std::vector<cv::DMatch> inliers;
  	std::vector<double> tr_delta = findInliers(kpts1, kpts2, matches, inliers, false, false);
    time_point t4 = std::chrono::steady_clock::now();

    // debug matches
    // cv::Mat matchImg;
    // // cv::drawMatches(im1, kpts1, im2, kpts2, inliers, matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1));
    // cv::drawMatches(im1, kpts1, im2, kpts2, matches, matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1));
    // cv::namedWindow("Matches", cv::WINDOW_NORMAL);
    // cv::imshow("Matches", matchImg);
    // cv::waitKey(0);

    // compute relative depth
    cv::Mat relDepthMap;
    depthanything_cls->compute(im1, relDepthMap);
    depthMap = relDepthMap;
    time_point t5 = std::chrono::steady_clock::now();

    // compute sparse depth prior map and mask
    cv::Mat sparseDepthMap = cv::Mat::zeros(relDepthMap.rows, relDepthMap.cols, CV_32F);
    cv::Mat sparseDepthMask = cv::Mat::zeros(relDepthMap.rows, relDepthMap.cols, CV_32F);
    int num_pts = fillSparseDepthMap(kpts1, kpts2, inliers, sparseDepthMap, sparseDepthMask, true);
    if (num_pts < 3) {
      cout << "Not enough points for metric depth estimation" << endl;
      return 0;
    }

    // compute metric depth by scale and shift
    ArrayXXf estimate = Map<ArrayXXf>(relDepthMap.ptr<float>(), relDepthMap.rows, relDepthMap.cols);
    // estimate = estimate.inverse(); // invert to metric depth
    ArrayXXf target   = Map<ArrayXXf>(sparseDepthMap.ptr<float>(), sparseDepthMap.rows, sparseDepthMap.cols);
    ArrayXXf valid    = Map<ArrayXXf>(sparseDepthMask.ptr<float>(), sparseDepthMask.rows, sparseDepthMask.cols);

    LeastSquaresEstimator estimator(estimate, target, valid);
    estimator.compute_scale_and_shift();
    estimator.apply_scale_and_shift();
    estimator.clamp_min_max(min_depth, max_depth, true);

    ArrayXXf output = estimator.get_output();
    output = output.inverse(); // invert to metric depth
    depthMap = cv::Mat(output.rows(), output.cols(), CV_32F, const_cast<float*>(output.data())).clone();

    time_point t6 = std::chrono::steady_clock::now();

    // debug error checking
    // float error = (valid.select(output, 0.0f) - valid.select(target.inverse(), 0.0f)).abs().sum() / valid.sum(); // inverted depth
    // float error = (valid.select(output, 0.0f) - valid.select(target, 0.0f)).abs().sum() / valid.sum();
    // cout << "mean depth error: " << error << endl;

    // performance timers
    double text = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
    double tmatch = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
    double tfilt = std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count();
    double tdepth = std::chrono::duration_cast<std::chrono::duration<double>>(t5 - t4).count();
    double tdepthmetric = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t5).count();
    double ttot = std::chrono::duration_cast<std::chrono::duration<double>>(t6 - t1).count();

    cout << "Number of inliers: " << inliers.size() << endl;
    cout << "Total time: " << ttot << endl;
    cout << "extract: " << text << ", match: " << tmatch << ", filter: " << tfilt << ", relative depth: " << tdepth << ", metric depth: " << tdepthmetric << endl;

    // // // plot depth image
    // // double minVal, maxVal;
    // // cv::minMaxLoc(depthMap, &minVal, &maxVal);
    // // cout << "Max depth: " << maxVal << ", Min depth: " << minVal << endl;
    // // // cv::Mat depthViz = sparseDepthMap.clone() * 255.0 / maxVal;
    // // cv::Mat depthViz = depthMap.clone() * 255.0 / maxVal;
    // // depthViz.convertTo(depthViz, CV_8U);
    // // cv::Mat depthColor;
    // // cv::applyColorMap(depthViz, depthColor, cv::COLORMAP_JET);
    // // cv::namedWindow("Depth", cv::WINDOW_NORMAL);
    // // cv::imshow("Depth", depthColor);

    // // cv::namedWindow("Image", cv::WINDOW_NORMAL);
    // // cv::imshow("Image", im1);
    // // cv::waitKey(0);

  	return inliers.size();
  }

  int DepthPipe::fillSparseDepthMap(const std::vector<cv::KeyPoint> &kpts_left, const std::vector<cv::KeyPoint> &kpts_right,
    const std::vector<cv::DMatch> &matches, cv::Mat &depth_map, cv::Mat &depth_mask, bool inverted)
  {
    // Input depth map should be correct size
    depth_map = cv::Mat::zeros(depth_map.rows, depth_map.cols, CV_32F);
    depth_mask = cv::Mat::zeros(depth_map.rows, depth_map.cols, CV_32F);

    if (kpts_left.empty() || kpts_right.empty() || matches.empty()) {
        std::cerr << "Input keypoints or matches are empty!" << std::endl;
        return 0;
    }

    std::vector<cv::Point3f> depth_points;

    for (const auto& match : matches) {
        // Get coordinates of matched keypoints
        const cv::KeyPoint& kp_left = kpts_left[match.queryIdx];
        const cv::KeyPoint& kp_right = kpts_right[match.trainIdx];

        // Compute disparity (difference in x-coordinates of matched keypoints)
        float disparity = kp_left.pt.x - kp_right.pt.x;
        if (disparity <= 0.0f) continue; // Ignore negative or zero disparities

        // Compute depth using the disparity
        float depth = (focal_length * baseline) / disparity;
        if (depth > max_depth || depth < min_depth) continue; // Ignore depths outside the range
        if (inverted) depth = 1.0f / depth;

        // Add depth information at the matched point location
        depth_points.emplace_back(kp_left.pt.x, kp_left.pt.y, depth);
    }

    if (depth_points.empty()) {
        return 0;
    }

    // Generate the sparse depth map
    for (auto depth_pt : depth_points) {
      int x = static_cast<int>(depth_pt.x);
      int y = static_cast<int>(depth_pt.y);
      depth_map.at<float>(y, x) = depth_pt.z;
      depth_mask.at<float>(y, x) = 1.0;
    }

    return depth_points.size();
  }

  void DepthPipe::computeFeaturesCUDA(const cv::Mat& I1, const cv::Mat& I2,
    std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc1, std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc2)
  {

    cv::Mat gI1, gI2;
    bool isRGB = false;

    if(I1.channels()==3)
    {
        if(isRGB)
        {
            cvtColor(I1,gI1,cv::COLOR_RGB2GRAY);
            cvtColor(I2,gI2,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(I1,gI1,cv::COLOR_BGR2GRAY);
            cvtColor(I2,gI2,cv::COLOR_BGR2GRAY);
        }
    }
    else if(I1.channels()==4)
    {
        if(isRGB)
        {
            cvtColor(I1,gI1,cv::COLOR_RGBA2GRAY);
            cvtColor(I2,gI2,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(I1,gI1,cv::COLOR_BGRA2GRAY);
            cvtColor(I2,gI2,cv::COLOR_BGRA2GRAY);
        }
    }

    gI1.convertTo(gI1, CV_32FC1);
    gI2.convertTo(gI2, CV_32FC1);

    float scale = std::min(float(sift_max_dim)/float(gI1.cols), float(sift_max_dim)/float(gI1.rows));
    if (scale >= 1.0) scale = 1.0;
    else {
      cv::resize(gI1, gI1, cv::Size(), scale, scale, cv::INTER_LINEAR);
      cv::resize(gI2, gI2, cv::Size(), scale, scale, cv::INTER_LINEAR);
    }

    // float initBlur = 1.6f;
    // float thresh = 1.2f; // for stereo
    // float thresh = 1.0f; // for hybrid
    float initBlur = 1.0f;
    float thresh = 2.0f;

    CudaImage img1, img2;
    // don't include memory allocation in timeing as this can in theory be done once
    img1.Allocate(gI1.cols, gI1.rows, iAlignUp(gI1.cols, 128), false, NULL, (float*)gI1.data);
    img2.Allocate(gI2.cols, gI2.rows, iAlignUp(gI2.cols, 128), false, NULL, (float*)gI2.data);

    img1.Download();
    img2.Download();

    float *memoryTmpCUDA = AllocSiftTempMemory(I1.cols, I1.rows, 5, true);
    // float *memoryTmpCUDA = AllocSiftTempMemory(I1.cols, I1.rows, 5, false);
    // ExtractSift(siftdata_1, img1, 5, initBlur, thresh, 0.0f, false, memoryTmpCUDA);
    ExtractSift(siftdata_1, img1, 5, initBlur, thresh, 0.0f, true, memoryTmpCUDA, 1.0f/scale);

    // memoryTmpCUDA = AllocSiftTempMemory(I2.cols, I2.rows, 5, false);
    // ExtractSift(siftdata_2, img2, 5, initBlur, thresh, 0.0f, false, memoryTmpCUDA);
    ExtractSift(siftdata_2, img2, 5, initBlur, thresh, 0.0f, true, memoryTmpCUDA, 1.0f/scale);

    FreeSiftTempMemory(memoryTmpCUDA);

    CUDAtoCV(siftdata_1, siftdata_2, kpts1, desc1, kpts2, desc2);
  }

  void DepthPipe::CUDAtoCV(SiftData &siftdata1, SiftData &siftdata2, std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc1,
    std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc2)
  {
    #ifdef MANAGEDMEM
      SiftPoint *sift1 = siftdata1.m_data;
      SiftPoint *sift2 = siftdata2.m_data;
    #else
      SiftPoint *sift1 = siftdata1.h_data;
      SiftPoint *sift2 = siftdata2.h_data;
    #endif

    kpts1.clear();
    kpts2.clear();
    desc1 = cv::Mat(siftdata1.numPts, 128, CV_32F);
    desc2 = cv::Mat(siftdata2.numPts, 128, CV_32F);

    cv::Mat row;
    for (int32_t i=0; i<siftdata1.numPts; i++) {
      kpts1.push_back(cv::KeyPoint(sift1[i].xpos, sift1[i].ypos, sift1[i].scale, sift1[i].orientation, 1, (int32_t)log2(sift1[i].subsampling)));
      // int32_t octave = (int32_t)log2(sift1[i].subsampling);
      row = desc1.row(i);
      std::memcpy(row.data, sift1[i].data, 128*sizeof(float));
    }

    for (int32_t i=0; i<siftdata2.numPts; i++) {
      kpts2.push_back(cv::KeyPoint(sift2[i].xpos, sift2[i].ypos, sift2[i].scale, sift2[i].orientation, 1, (int32_t)log2(sift2[i].subsampling)));
      // int32_t octave = (int32_t)log2(sift2[i].subsampling);
      row = desc2.row(i);
      std::memcpy(row.data, sift2[i].data, 128*sizeof(float));
    }
  }

  void DepthPipe::matchCUDA(std::vector<cv::KeyPoint> &kpts1, cv::Mat &desc1,
    std::vector<cv::KeyPoint> &kpts2, cv::Mat &desc2, std::vector<cv::DMatch> &matches,
    const float ratio_thresh, const bool cross_check)
  {
    // Copy keypoints to cudaSift objects
    #ifdef MANAGEDMEM
      SiftPoint *sift1 = siftdata_1.m_data;
      SiftPoint *sift2 = siftdata_2.m_data;
    #else
      SiftPoint *sift1 = siftdata_1.h_data;
      SiftPoint *sift2 = siftdata_2.h_data;
    #endif

    cv::Mat row;
    for (int32_t i=0; i<kpts1.size(); i++) {
      row = desc1.row(i);
      std::memcpy(sift1[i].data, row.data, 128*sizeof(float));
      sift1[i].xpos = kpts1[i].pt.x;
      sift1[i].ypos = kpts1[i].pt.y;
      sift1[i].orientation = kpts1[i].angle;
    }
    siftdata_1.numPts = kpts1.size();
    CopyToDevice(siftdata_1);

    for (int32_t i=0; i<kpts2.size(); i++) {
      row = desc2.row(i);
      std::memcpy(sift2[i].data, row.data, 128*sizeof(float));
      sift2[i].xpos = kpts2[i].pt.x;
      sift2[i].ypos = kpts2[i].pt.y;
      sift2[i].orientation = kpts2[i].angle;
    }
    siftdata_2.numPts = kpts2.size();
    CopyToDevice(siftdata_2);

    // BF matching with left right consistency
    int32_t i1c,i2c,i1c2;

    matches.clear();

    MatchSiftData(siftdata_1, siftdata_2);
    if (cross_check)
      MatchSiftData(siftdata_2, siftdata_1);

    std::vector<float> score_match(kpts2.size()*2, 0.0);
    std::vector<int> index_match(kpts2.size()*2, -1);

    int index;
    for (int32_t i1c=0; i1c<siftdata_1.numPts; i1c++) {
      // check distance ratio thresh
      if (sift1[i1c].ambiguity > ratio_thresh)
        continue;
      i2c = sift1[i1c].match;

      // check epipolar constraint
      if (fabs(sift1[i1c].match_ypos - sift1[i1c].ypos) > (*f_settings)["Matcher.match_disp_tolerance"].real())
        continue;
      // filter negative disparities
      if (sift1[i1c].xpos<=sift2[i2c].xpos)
        continue;

      // check mutual best feature match in both images
      if (cross_check) {
        i1c2 = sift2[i2c].match;
        if (i1c != i1c2)
          continue;
      }

      // return matches where each trainIdx index is associated with only one queryIdx 
      // trainIdx has not been matched yet
      if (score_match[i2c] == 0.0) {
        score_match[i2c] = sift1[i1c].score;
        matches.push_back(cv::DMatch(i1c, i2c, sift1[i1c].score));
        index_match[i2c] = matches.size() - 1;
      }
      // we have already a match for trainIdx: if stored match is worse => replace it
      else if (sift1[i1c].score > score_match[i2c]) {
        index = index_match[i2c];
        assert(matches[index].trainIdx == i2c);
        matches[index].queryIdx = i1c;
        matches[index].distance = sift1[i1c].score;
      }
    }
  }

  std::vector<double> DepthPipe::findInliers(const std::vector<cv::KeyPoint> &kpts1, const std::vector<cv::KeyPoint> &kpts2, 
    const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inliers, bool is_fish1, bool is_fish2)
  {
    inliers.clear();
    if (matches.size()<10)
      return vector<double>({0,0,0,0,0,0});

    // copy matched points into cv vector
    std::vector<cv::Point2d> points1(matches.size());
    std::vector<cv::Point2d> points2(matches.size());

    for(size_t i = 0; i < matches.size(); i++) {
      int idx1, idx2;
      idx1 = matches[i].queryIdx;
      idx2 = matches[i].trainIdx;
      points1[i] = cv::Point2f(kpts1[idx1].pt.x, kpts1[idx1].pt.y);
      points2[i] = cv::Point2f(kpts2[idx2].pt.x, kpts2[idx2].pt.y);
    }

    // find essential matrix with 5-point algorithm
    double focal = (*f_settings)["Stereo.f"].real();
    cv::Point2d pp((*f_settings)["Stereo.cx"].real(),(*f_settings)["Stereo.cy"].real());
    float prob = 0.999;
    float thresh = 1.0; // projection error is measured by projection onto stereo
    int method = cv::RANSAC;
    cv::Mat mask;
    cv::Mat essential_mat = cv::findEssentialMat(points1, points2, focal, pp, method, prob, thresh, mask);

    // recover motion from essential matrix
    cv::Mat t;
    cv::Mat R;
    int num_inliers = cv::recoverPose(essential_mat, points1, points2, R, t, focal, pp, mask);

    // if (num_inliers < 10)
      // return vector<double>({0,0,0,0,0,0,0});

    for(int i = 0; i < mask.rows; i++) {
      if(mask.at<unsigned char>(i)){
        inliers.push_back(matches[i]);
      }
    }

    // compute rotation angles
    // double ry = asin(R.at<double>(0,2));
    // double rx = asin(-R.at<double>(1,2)/cos(ry));
    // double rz = asin(-R.at<double>(0,1)/cos(ry));

    vector<double> q_delta = toQuaternion(R);

    double t_mag = sqrtf(t.at<double>(0)*t.at<double>(0) + t.at<double>(1)*t.at<double>(1) + t.at<double>(2)*t.at<double>(2));
    
    // return parameter vector
    vector<double> tr_delta;
    tr_delta.resize(7);
    // x,y,z,qw,qx,qy,qz
    tr_delta[0] = t.at<double>(0)/t_mag;
    tr_delta[1] = t.at<double>(1)/t_mag;
    tr_delta[2] = t.at<double>(2)/t_mag;
    tr_delta[3] = q_delta[0];
    tr_delta[4] = q_delta[1];
    tr_delta[5] = q_delta[2];
    tr_delta[6] = q_delta[3];

    return tr_delta;
  }

  std::vector<double> DepthPipe::toQuaternion(const cv::Mat &M)
  {
      Eigen::Matrix<double,3,3> eigMat = toMatrix3d(M);
      Eigen::Quaterniond q(eigMat);

      std::vector<double> v(4);
      v[0] = q.w();
      v[1] = q.x();
      v[2] = q.y();
      v[3] = q.z();

      return v;
  }

  Eigen::Matrix<double,3,3> DepthPipe::toMatrix3d(const cv::Mat &cvMat3)
  {
      Eigen::Matrix<double,3,3> M;
  
      M << cvMat3.at<double>(0,0), cvMat3.at<double>(0,1), cvMat3.at<double>(0,2),
           cvMat3.at<double>(1,0), cvMat3.at<double>(1,1), cvMat3.at<double>(1,2),
           cvMat3.at<double>(2,0), cvMat3.at<double>(2,1), cvMat3.at<double>(2,2);
  
      return M;
  }

} // namespace depthpipe
