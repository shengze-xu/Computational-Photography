#pragma once
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

namespace std
{
	template<> struct hash<cv::Vec3i>
	{
		std::size_t operator()(const cv::Vec3i & s) const noexcept
		{
			std::size_t seed = s.channels;
			for (int i = 0; i < s.channels; i++)
				seed ^= s[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};
}

struct Camera
{
	double camera_factor;
	double fx_rgb;
	double fy_rgb;
	double cx_rgb;
	double cy_rgb;
};


class SpacePoint {
public:
	unsigned int i, j, d;
	double x1, y1, z1; //before transform
	double x2, y2, z2; //after transform
	double nx, ny, nz;
	double tx, ty, tz;
	uchar r, g, b;
	std::vector<double> confidences;
	double confidence;
	bool star;
	bool low_depth_confidence;
	bool low_rgb_confidence;

	bool operator<(const SpacePoint sp)
	{
		return confidence < sp.confidence;
	}
};

class Image
{
public:
	std::string RGB_PATH;
	std::string DEPTH_PATH;
	std::string DEPTH_FROM_RGB_PATH;
	double QW,QX,QY,QZ;
	double TX,TY,TZ;
	cv::Mat T;
	std::vector<SpacePoint> pointcloud;

	void generatePointCloud(std::string path, Camera camera);
	void parametersToTransformMatrix();
	void writecloud(std::string filename);
	void calculateConfidence(Image &ref, std::string path, Camera camera);
	void denoiseAccordingToDepth(Image &ref, std::string path, Camera camera);
private:
	cv::Mat setFrequencyUseVarience(cv::Mat depth);
	cv::Mat setFrequencyUseSobel(cv::Mat depth);
	cv::Mat setFrequencyUseLaplacian(cv::Mat depth);
};

class TestSet {
public:
	struct Camera camera;
	std::string path;
	std::vector<Image> images;
	std::vector<int> indexs;
	std::list<SpacePoint> cloud;
	std::vector<SpacePoint> cloudofeveryimage;
	std::unordered_map<cv::Vec3i, SpacePoint*> hashcloud;

	TestSet(std::string path, int num,int front,int debug = 0);
	
	void generatePointCloud();
	void calculateConfidenceAndAddToCloud(int range,double ratio);
	void writeRawImages(std::string filename);
	//void writeCloud(std::string filename, double ratio);
	//void writeCloudEveryImage(std::string filename);
	//void writeCloudEveryImagePLY(std::string filename);
	void denoiseAccordingToDepth(int range);
	void removeduplicate();
	void writeHashCloud(std::string filename);
	void generateNormal();
	void writeNormal(std::string filename);
private:
	void loadImageInformation(int num,int front,int select);
};