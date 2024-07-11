#pragma comment(lib,"pthreadVC2.lib")
#include <iostream>
#include <stdio.h>
#include "construction.h"
#include <time.h>
#include "kdtree.hpp"
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <pthread.h>

//namespace std
//{
//	template<> struct hash<cv::Vec3i>
//	{
//		std::size_t operator()(const cv::Vec3i & s) const noexcept
//		{
//			std::size_t seed = s.channels;
//			for (int i = 0; i < s.channels; i++)
//				seed ^= s[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//			return seed;
//		}
//	};
//}

using namespace std;
//using namespace Kdtree;
using namespace cv;

void print_nodes(const Kdtree::KdNodeVector &nodes) {
	size_t i, j;
	for (i = 0; i < nodes.size(); ++i) {
		if (i > 0)
			cout << " ";
		cout << "(";
		for (j = 0; j < nodes[i].point.size(); j++) {
			if (j > 0)
				cout << ",";
			cout << nodes[i].point[j];
		}
		cout << ")";
	}
	cout << endl;
}

pthread_mutex_t mutex;
void *print_msg(void *arg) {
	int i = 0;
	for (i = 0; i < 15; i++) {
		printf("output : %d\n", i);
	}
	return NULL;
}


int main(int argc, char** argv)
{
	/*pthread_t id1;
	pthread_t id2;
	pthread_create(&id1, NULL, print_msg, NULL);
	pthread_create(&id2, NULL, print_msg, NULL);
	pthread_join(id1, NULL);
	pthread_join(id2, NULL);
	return 0;*/
	string t = "plant";

	string path("E:/data_plant/");
	TestSet testset(path,50,-1,0);
	testset.generatePointCloud();
	//testset.denoiseAccordingToDepth(1);
	testset.writeRawImages("result/afterFusion" + t + ".txt");
	testset.calculateConfidenceAndAddToCloud(1,0.5);
	testset.removeduplicate();
	testset.writeHashCloud("result/afterReduceDuplication"+ t +".txt");
	testset.generateNormal();
	testset.writeNormal("result/normal" + t + ".txt");



	//vector<SpacePoint> pointcloud_sphere;
	//FILE* file;
	//fopen_s(&file, "D:/sphere2.txt", "r");
	//SpacePoint sp;
	//while (fscanf_s(file, "%lf", &sp.x2) > 0)
	//{
	//	fscanf_s(file, "%lf", &sp.y2);
	//	fscanf_s(file, "%lf", &sp.z2);
	//	pointcloud_sphere.push_back(sp);
	//}
	//fclose(file);

	//Kdtree::KdNodeVector nodes;
	//for (int i = 0; i < pointcloud_sphere.size(); i++)
	//{
	//	vector<double> point(3);
	//	point[0] = (pointcloud_sphere[i]).x2;
	//	point[1] = (pointcloud_sphere[i]).y2;
	//	point[2] = (pointcloud_sphere[i]).z2;
	//	nodes.push_back(point);
	//}
	//Kdtree::KdTree tree(&nodes);

	//for (int i = 0; i < pointcloud_sphere.size(); i++)
	//{
	//	vector<double> point(3);
	//	point[0] = (pointcloud_sphere[i]).x2;
	//	point[1] = (pointcloud_sphere[i]).y2;
	//	point[2] = (pointcloud_sphere[i]).z2;
	//	Kdtree::KdNodeVector result;
	//	tree.k_nearest_neighbors(point, 10, &result);
	//	double x_sum = point[0];
	//	double y_sum = point[1];
	//	double z_sum = point[2];
	//	for (int j = 0; j < result.size(); j++)
	//	{
	//		x_sum += result[j].point[0];
	//		y_sum += result[j].point[1];
	//		z_sum += result[j].point[2];
	//	}
	//	double x_avg = x_sum / (result.size() + 1);
	//	double y_avg = y_sum / (result.size() + 1);
	//	double z_avg = z_sum / (result.size() + 1);
	//	Mat A = Mat::zeros(3, 3, CV_64FC1);
	//	for (int j = 0; j < result.size(); j++)
	//	{
	//		A.at<double>(0, 0) += (result[j].point[0] - x_avg)*(result[j].point[0] - x_avg);
	//		A.at<double>(0, 1) += (result[j].point[0] - x_avg)*(result[j].point[1] - y_avg);
	//		A.at<double>(0, 2) += (result[j].point[0] - x_avg)*(result[j].point[2] - z_avg);
	//		A.at<double>(1, 0) += (result[j].point[1] - y_avg)*(result[j].point[0] - x_avg);
	//		A.at<double>(1, 1) += (result[j].point[1] - y_avg)*(result[j].point[1] - y_avg);
	//		A.at<double>(1, 2) += (result[j].point[1] - y_avg)*(result[j].point[2] - z_avg);
	//		A.at<double>(2, 0) += (result[j].point[2] - z_avg)*(result[j].point[0] - x_avg);
	//		A.at<double>(2, 1) += (result[j].point[2] - z_avg)*(result[j].point[1] - y_avg);
	//		A.at<double>(2, 2) += (result[j].point[2] - z_avg)*(result[j].point[2] - z_avg);
	//	}
	//	cv::Mat eValuesMat;
	//	cv::Mat eVectorsMat;
	//	eigen(A, eValuesMat, eVectorsMat);
	//	int idx_min;
	//	double eValue_min = DBL_MAX;
	//	for (int k = 0; k < 3; k++)
	//	{
	//		if (eValuesMat.at<double>(k, 0) <= eValue_min)
	//		{
	//			eValue_min = eValuesMat.at<double>(k, 0);
	//			idx_min = k;
	//		}
	//	}
	//	/*cout << A << endl;
	//	cout << eValuesMat << endl;
	//	cout << eVectorsMat << endl;
	//	cout << idx_min << endl;
	//	cout << eVectorsMat.at<double>(idx_min, 0) << endl;
	//	cout << eVectorsMat.at<double>(idx_min, 1) << endl;
	//	cout << eVectorsMat.at<double>(idx_min, 2) << endl;*/
	//	pointcloud_sphere[i].nx = eVectorsMat.at<double>(idx_min, 0);
	//	pointcloud_sphere[i].ny = eVectorsMat.at<double>(idx_min, 1);
	//	pointcloud_sphere[i].nz = eVectorsMat.at<double>(idx_min, 2);
	//}
	//fopen_s(&file, "D:/sphereoutput.txt", "w");
	//for (int i = 0; i < pointcloud_sphere.size(); i++)
	//{
	//	if (pointcloud_sphere[i].x2+ pointcloud_sphere[i].y2+pointcloud_sphere[i].z2 <= 0)
	//	{
	//		fprintf(file, "%lf %lf %lf %lf %lf %lf\n",
	//			pointcloud_sphere[i].x2,
	//			pointcloud_sphere[i].y2,
	//			pointcloud_sphere[i].z2,
	//			pointcloud_sphere[i].nx,
	//			pointcloud_sphere[i].ny,
	//			pointcloud_sphere[i].nz
	//			);
	//	}
	//	else
	//	{
	//		fprintf(file, "%lf %lf %lf %lf %lf %lf\n",
	//			pointcloud_sphere[i].x2,
	//			pointcloud_sphere[i].y2,
	//			pointcloud_sphere[i].z2,
	//			-pointcloud_sphere[i].nx,
	//			-pointcloud_sphere[i].ny,
	//			-pointcloud_sphere[i].nz
	//		);
	//	}
	//	
	//}
	return 0;
}
