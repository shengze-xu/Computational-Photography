#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

class DRPicture
{
public:
	std::string rgb_path;
	std::string depth_path;
	cv::Mat rgb;
	cv::Mat disparity;
	cv::Mat P;
	cv::Mat R;
	cv::Mat R_T;
	cv::Mat T;
	cv::Mat K;
	cv::Mat K_1;
	double QW, QX, QY, QZ;
	double TX, TY, TZ;
	double fx_rgb;
	double fy_rgb;
	double cx_rgb;
	double cy_rgb;

	void generateP();
	void generateR();
	void generateT();
	void generateK();
};

class DepthRecoverSolution
{
public:
	std::string path;
	double fx_rgb, fy_rgb, cx_rgb, cy_rgb;
	std::vector<DRPicture> pictures;
	std::vector<int> indexs;

	void loadInfo(std::string p);
	void loadRGBImgAndCreateDisparity();
	void disparityCalculation();
	void outputdepth(std::string p);
};