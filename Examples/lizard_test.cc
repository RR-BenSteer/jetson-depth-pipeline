#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <string>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "DepthPipe.h"

namespace fs = std::filesystem;
using namespace std;
using namespace depthpipe;

void readBinaryDepth(const std::string& path, cv::Mat &depth);

void readImagePaths(const string &base_dir, vector<string> &left_image_paths,
                vector<string> &right_image_paths, vector<string> &depth_image_paths);

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
    readImagePaths(string(argv[2]), left_image_paths, right_image_paths, depth_image_paths);

    const int nImages = left_image_paths.size();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    cv::Mat imLeft, imRight;

    DepthPipe depthpipe(argv[1]);

    // Main loop
    for(int ni=0; ni<nImages; ni++)
    {
        cout << "image " << ni << " of " << nImages << endl;   
        imLeft = cv::imread(left_image_paths[ni],cv::IMREAD_UNCHANGED);
        imRight = cv::imread(right_image_paths[ni],cv::IMREAD_UNCHANGED);

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(left_image_paths[ni]) << endl;
            return 1;
        }
        if(imRight.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(right_image_paths[ni]) << endl;
            return 1;
        }

        cv::Mat depthMap;
        depthpipe.process(imLeft, imRight, depthMap);

        cv::Mat gtDepth;
        readBinaryDepth(depth_image_paths[ni], gtDepth);

        // Compute RMSE with gt depth
        double rmse = computeDepthRMSE(depthMap, gtDepth, 1.0, 2.0);
        cout << "RMSE: " << rmse << endl;

        // plot images
        // double minVal, maxVal;
        // cv::minMaxLoc(depthMap, &minVal, &maxVal);
        // // cout << "Max pred depth: " << maxVal << ", Min pred depth: " << minVal << endl;
        // cv::Mat depthViz = depthMap.clone() * 255.0 / 3.0;
        // depthViz.convertTo(depthViz, CV_8U);
        // cv::Mat depthColor;
        // cv::applyColorMap(depthViz, depthColor, cv::COLORMAP_JET);
        // cv::namedWindow("Pred Depth", cv::WINDOW_NORMAL);
        // cv::imshow("Pred Depth", depthColor);

        // cv::minMaxLoc(gtDepth, &minVal, &maxVal);
        // // cout << "Max gt depth: " << maxVal << ", Min gt depth: " << minVal << endl;
        // cv::Mat depthViz2 = gtDepth.clone() * 255.0 / 3.0;
        // depthViz2.convertTo(depthViz2, CV_8U);
        // cv::Mat depthColor2;
        // cv::applyColorMap(depthViz2, depthColor2, cv::COLORMAP_JET);
        // cv::namedWindow("GT Depth", cv::WINDOW_NORMAL);
        // cv::imshow("GT Depth", depthColor2);

        // cv::namedWindow("Image", cv::WINDOW_NORMAL);
        // cv::imshow("Image", imLeft);
        // cv::waitKey(0);
    }

    return 0;
}

void readImagePaths(const string &base_dir, vector<string> &left_image_paths,
                vector<string> &right_image_paths, vector<string> &depth_image_paths)
{
    string left_folder = base_dir + "/images/left/";
    string right_folder = base_dir + "/images/right/";
    string depth_folder = base_dir + "/depth/";

    std::unordered_map<std::string, std::string> left_map;
    std::unordered_map<std::string, std::string> right_map;
    std::unordered_map<std::string, std::string> depth_map;

    // Step 1: Store the full path of all images in the left folder
    for (const auto& entry : fs::directory_iterator(left_folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().stem().string(); // Get the base name without extension
            left_map[filename] = entry.path().string(); // Map base name to full path
        }
    }

    // Step 2: Store the full path of all images in the right folder
    for (const auto& entry : fs::directory_iterator(right_folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().stem().string(); // Get the base name without extension
            right_map[filename] = entry.path().string(); // Map base name to full path
        }
    }

    // Step 3: Store the full path of all images in the depth folder
    for (const auto& entry : fs::directory_iterator(depth_folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().stem().string(); // Get the base name without extension
            depth_map[filename] = entry.path().string(); // Map base name to full path
        }
    }

    // Step 4: Check for matching images across all three folders
    std::vector<std::string> common_filenames;
    for (const auto& [base_name, _] : left_map) {
        if (right_map.count(base_name) && depth_map.count(base_name)) {
            common_filenames.push_back(base_name);
        }
    }

    // Step 5: Sort the common filenames numerically
    std::sort(common_filenames.begin(), common_filenames.end(), [](const std::string& a, const std::string& b) {
        return std::stoi(a) < std::stoi(b);
    });

    // Step 6: Store the sorted paths in the output vectors
    for (const auto& filename : common_filenames) {
        left_image_paths.push_back(left_map[filename]);
        right_image_paths.push_back(right_map[filename]);
        depth_image_paths.push_back(depth_map[filename]);
    }

}

void readBinaryDepth(const std::string& path, cv::Mat &depth) {
    // Open file in binary mode
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read the first line to get width, height, and channels
    std::string header;
    std::getline(file, header);

    std::istringstream headerStream(header);
    std::string token;
    int width, height, channels;
    char delimiter = '&';
    
    // Parse width, height, channels from the first line
    std::getline(headerStream, token, delimiter);
    width = std::stoi(token);

    std::getline(headerStream, token, delimiter);
    height = std::stoi(token);

    std::getline(headerStream, token, delimiter);
    channels = std::stoi(token);

    // Move the file read pointer past the delimiter section
    file.seekg(0, std::ios::beg);
    int numDelimiters = 0;
    char byte;
    while (file.get(byte)) {
        if (byte == '&') {
            numDelimiters++;
        }
        if (numDelimiters >= 3) {
            break;
        }
    }

    // Read the remaining binary data into a vector
    std::vector<float> data;
    std::streampos pos = file.tellg();

    // Move to the end to calculate remaining file size
    file.seekg(0, std::ios::end);
    std::streamsize remainingBytes = file.tellg() - pos;
    file.seekg(pos);

    // Ensure the remaining data size is a multiple of sizeof(float)
    if (remainingBytes % sizeof(float) != 0) {
        throw std::runtime_error("Data size after delimiters is not a multiple of float size.");
    }

    // Read the remaining binary data into a vector of floats
    std::size_t numElements = remainingBytes / sizeof(float);
    data.resize(numElements);

    file.read(reinterpret_cast<char*>(data.data()), remainingBytes);
    if (!file) {
        throw std::runtime_error("Error reading float data from file.");
    }

    // float value;
    // while (file.read(reinterpret_cast<char*>(&value), sizeof(float))) {
    //     data.push_back(value);
    // }

    // Close the file
    file.close();

    int total_elements = width * height * channels;
    cout << "width: " << width << ", height: " << height << ", channels: " << channels << endl;
    cout << "total elements: " << total_elements << endl;
    cout << "data size: " << data.size() << endl;
    if (data.size() != total_elements) {
        throw std::runtime_error("Data size mismatch");
    }

    // Convert the vector to a cv::Mat
    depth = cv::Mat(height, width, CV_32F, data.data()).clone();
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