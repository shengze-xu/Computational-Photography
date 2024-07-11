#include "construction.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <list>
#include <float.h>
#include "kdtree.hpp"

int count_false = 0;

using namespace std;
using namespace cv;

void Image::generatePointCloud(string path,struct Camera camera)
{
	cout << "generating point cloud:" << RGB_PATH << endl;
	Mat rgb = imread(path + RGB_PATH);
	Mat depth = imread(path + DEPTH_PATH, CV_16UC1);
	Mat sample(rgb.size(), CV_8UC1,Scalar(255));
	parametersToTransformMatrix();

	Mat frequency = setFrequencyUseLaplacian(depth);
	srand(0);
	for (int i = 0; i < rgb.rows; i++)
	{
		for (int j = 0; j < rgb.cols; j++)
		{
			
			SpacePoint sp;
			ushort d = depth.at<ushort>(i, j);
			if (d == 0) continue;
			ushort f = frequency.at<ushort>(i, j);
			if (rand() % 0x3FFF > f) continue;
			sample.at<uchar>(i, j) = 0;
			double z = (double)d / camera.camera_factor;
			double x = (j - camera.cx_rgb)*z / camera.fx_rgb;
			double y = (i - camera.cy_rgb)*z / camera.fy_rgb;
			sp.i = i;
			sp.j = j;
			sp.d = d;
			sp.x1 = x;
			sp.y1 = y;
			sp.z1 = z;
			Vec3b color = rgb.at<Vec3b>(i, j);
			Mat point_before_transform(4, 1, CV_64FC1);
			point_before_transform.at<double>(0, 0) = x;
			point_before_transform.at<double>(1, 0) = y;
			point_before_transform.at<double>(2, 0) = z;
			point_before_transform.at<double>(3, 0) = 1;
			Mat point_after_transform = T * point_before_transform;
			x = point_after_transform.at<double>(0, 0);
			y = point_after_transform.at<double>(1, 0);
			z = point_after_transform.at<double>(2, 0);
			sp.x2 = x;
			sp.y2 = y;
			sp.z2 = z;
			sp.r = color[2];
			sp.g = color[1];
			sp.b = color[0];
			sp.tx = TX;
			sp.ty = TY;
			sp.tz = TZ;
			sp.low_depth_confidence = false;
			sp.low_rgb_confidence = false;
			sp.star = false;
			/*printf("%d %d %d\n", i, j, d);
			cout << T << endl;
			printf("%lf,%lf,%lf\n", sp.x1, sp.y1, sp.z1);
			printf("%lf,%lf,%lf\n", sp.x2, sp.y2, sp.z2);*/
			pointcloud.push_back(sp);
		}
	}
	imshow("sample", sample);
	waitKey(0);
	destroyAllWindows();
	
}

void Image::parametersToTransformMatrix()
{
	T = Mat(4, 4, CV_64FC1);
	T.at<double>(0, 0) = 1 - 2 * QY * QY - 2 * QZ * QZ;
	T.at<double>(0, 1) = 2 * QX * QY - 2 * QW * QZ;
	T.at<double>(0, 2) = 2 * QX * QZ + 2 * QW * QY;
	T.at<double>(1, 0) = 2 * QX * QY + 2 * QW * QZ;
	T.at<double>(1, 1) = 1 - 2 * QX * QX - 2 * QZ * QZ;
	T.at<double>(1, 2) = 2 * QY * QZ - 2 * QW * QX;
	T.at<double>(2, 0) = 2 * QX * QZ - 2 * QW * QY;
	T.at<double>(2, 1) = 2 * QY * QZ + 2 * QW * QX;
	T.at<double>(2, 2) = 1 - 2 * QX * QX - 2 * QY * QY;

	T.at<double>(0, 3) = TX;
	T.at<double>(1, 3) = TY;
	T.at<double>(2, 3) = TZ;

	T.at<double>(3, 0) = 0;
	T.at<double>(3, 1) = 0;
	T.at<double>(3, 2) = 0;
	T.at<double>(3, 3) = 1;
}

const int SAMLPlING_WINDOW_SIZE = 5;
const int HALF_SAMLPlING_WINDOW_SIZE = 2;
//variance range from 0 to 0x3fffffff
unsigned int VARIANCE_THRESHOLD = 0x0FFFFFFF;
unsigned int MIN_FREQUENCY = 0x0FFF;
unsigned int MAX_FREQUENCY = 0x5FFF;
/*return value range from 0 to 32767 */
Mat Image::setFrequencyUseVarience(Mat depth)
{
	Mat depth2 = depth.clone();
	Mat frequency = Mat::zeros(depth.size(), CV_16UC1);
	for (int i = 0; i < depth2.rows; i++)
	{
		for (int j = 0; j < depth2.cols; j++)
		{
			if (depth2.at<ushort>(i, j) == 0)
			{
				depth2.at<ushort>(i, j) = (ushort)0xFFFF;
			}
		}
	}

	for (int i = 0; i < depth2.rows; i++)
	{
		if (i > depth2.rows - SAMLPlING_WINDOW_SIZE) continue;
		for (int j = 0; j < depth2.cols; j++)
		{
			if (j > depth2.cols - SAMLPlING_WINDOW_SIZE) continue;
			int sum = 0;
			int square_sum = 0;
			for (int u = 0; u < SAMLPlING_WINDOW_SIZE; u++)
			{
				for (int v = 0; v < SAMLPlING_WINDOW_SIZE; v++)
				{
					ushort temp = depth2.at<ushort>(i + u, j + v);
					sum += temp;
					square_sum += temp * temp;
				}
			}
			double avg_sum = sum / SAMLPlING_WINDOW_SIZE / SAMLPlING_WINDOW_SIZE;
			double avg_square_sum = square_sum / SAMLPlING_WINDOW_SIZE / SAMLPlING_WINDOW_SIZE;
			int var = avg_square_sum - avg_sum * avg_sum;
			uint f = MAX_FREQUENCY;
			if (var >= VARIANCE_THRESHOLD)
			{
				f = MAX_FREQUENCY;
			}
			else
			{
				f = MIN_FREQUENCY + var * (MAX_FREQUENCY - MIN_FREQUENCY) / VARIANCE_THRESHOLD;
			}
			frequency.at<ushort>(i + HALF_SAMLPlING_WINDOW_SIZE, j + HALF_SAMLPlING_WINDOW_SIZE) = (ushort)f;
		}
	}
	return frequency;
}

uint GRADIENT_THRESHOLD = 0x1FFFF;
Mat Image::setFrequencyUseSobel(Mat depth)
{
	Mat depth2 = depth.clone();
	for (int i = 0; i < depth2.rows; i++)
	{
		for (int j = 0; j < depth2.cols; j++)
		{
			if (depth2.at<ushort>(i, j) == 0)
			{
				depth2.at<ushort>(i, j) = (ushort)0xFFFF;
			}
		}
	}
	Mat frequency = Mat::zeros(depth.size(), CV_16UC1);
	GaussianBlur(depth2, depth2, Size(5, 5), 0, 0);
	Mat Gx, Gy;
	Sobel(depth2, Gx, CV_32F, 1, 0, 5);
	Sobel(depth2, Gy, CV_32F, 0, 1, 5);
	Mat Gxy(depth.size(), CV_32F);
	double max_gxy = 0;
	for (int i = 0; i < depth2.rows; i++)
	{
		for (int j = 0; j < depth2.cols; j++)
		{
			double gx = Gx.at<float>(i, j);
			double gy = Gy.at<float>(i, j);
			double temp = sqrt(gx * gx + gy * gy);
			Gxy.at<float>(i, j) = temp;
			if (temp > max_gxy) max_gxy = temp;
		}
	}

	for (int i = 0; i < depth2.rows; i++)
	{
		for (int j = 0; j < depth2.cols; j++)
		{
			double gxy = Gxy.at<float>(i, j);
			int f;
			if (gxy >= GRADIENT_THRESHOLD)
			{
				f = MAX_FREQUENCY;
			}
			else
			{
				f = MIN_FREQUENCY + gxy * (MAX_FREQUENCY - MIN_FREQUENCY) / GRADIENT_THRESHOLD;
			}
			frequency.at<ushort>(i, j) = f;
		}
	}
	//imshow("f", frequency);

	return frequency;
}

const uint LAP_THRESHOLD = 0xFFF;
Mat Image::setFrequencyUseLaplacian(Mat depth)
{
	Mat depth2 = depth.clone();
	for (int i = 0; i < depth2.rows; i++)
	{
		for (int j = 0; j < depth2.cols; j++)
		{
			if (depth2.at<ushort>(i, j) == 0)
			{
				depth2.at<ushort>(i, j) = (ushort)0xFFFF;
			}
		}
	}
	Mat frequency = Mat::zeros(depth.size(), CV_16UC1);
	GaussianBlur(depth2, depth2, Size(3, 3), 0, 0);
	Mat lap;
	Laplacian(depth2, lap, CV_32F, 3, 1);
	int border = 20;
	for (int i = 0; i < depth2.rows; i++)
	{
		for (int j = 0; j < depth2.cols; j++)
		{
			if (i <= border || i >= depth2.rows - border || j <= border || j >= depth2.cols - 2 * border)
			{
				frequency.at<ushort>(i, j) = 0;
				continue;
			}
			double val = fabs(lap.at<float>(i, j));
			int f;
			if (val >= GRADIENT_THRESHOLD)
			{
				f = MAX_FREQUENCY;
			}
			else
			{
				f = MIN_FREQUENCY + val * (MAX_FREQUENCY - MIN_FREQUENCY) / LAP_THRESHOLD;
			}
			frequency.at<ushort>(i, j) = f;
			/*if (i % 10 == 0 && j % 10 == 0 && val!=0)
			{
				printf("%d,%d:%f\n", i, j, val);
			}*/
		}
	}
	/*imshow("f", frequency);
	waitKey(0);
	destroyAllWindows();*/
	return frequency;
}

TestSet::TestSet(string path, int num,int front,int debug)
{
	this->path = path;
	loadImageInformation(num,front,debug);
}

void TestSet::loadImageInformation(int num,int front,int select)
{
	FILE* file;
	double omit;
	char buffer[100];
	Image ii;

	fopen_s(&file, (path + "info/camera.txt").c_str(), "r");
	fscanf_s(file, "%lf", &camera.fx_rgb);
	fscanf_s(file, "%lf", &camera.fy_rgb);
	fscanf_s(file, "%lf", &camera.cx_rgb);
	fscanf_s(file, "%lf", &camera.cy_rgb);
	fscanf_s(file, "%lf", &camera.camera_factor);
	fclose(file);

	fopen_s(&file, (path + "info/associate_with_groundtruth.txt").c_str(), "r");
	while (fscanf_s(file, "%lf", &omit) > 0)
	{
		fscanf_s(file, "%s", buffer, 100);
		ii.RGB_PATH = buffer;
		fscanf_s(file, "%lf", &omit);
		fscanf_s(file, "%s", buffer, 100);
		ii.DEPTH_PATH = buffer;
		fscanf_s(file, "%lf", &omit);
		fscanf_s(file, "%lf", &ii.TX);
		fscanf_s(file, "%lf", &ii.TY);
		fscanf_s(file, "%lf", &ii.TZ);
		fscanf_s(file, "%lf", &ii.QX);
		fscanf_s(file, "%lf", &ii.QY);
		fscanf_s(file, "%lf", &ii.QZ);
		fscanf_s(file, "%lf", &ii.QW);
		images.push_back(ii);
	}
	fclose(file);

	if (front > images.size() || front <= 0)
	{
		front = images.size();
	}
	int step;
	if ( front < num) step = 1;
	else
	{
		step = front / num ;
	}
	if (step < 1) step = 1;
	for (int i = 0; i < front; i += step)
	{
		indexs.push_back(i);
	}
	//indexs.push_back(select);
}

void TestSet::generatePointCloud()
{
	for (int i = 0; i < indexs.size(); i++)
	{
		int idx = indexs[i];
		images[idx].generatePointCloud(path,camera);

	}
}

void TestSet::calculateConfidenceAndAddToCloud(int range, double ratio) 
{
	if (ratio <= 0.01) ratio = 0.01;
	if (ratio > 1) ratio = 1;
	int begin, end;
	for (int i = 0; i < indexs.size(); i++)
	{
		begin = i - range;
		end = i + range;
		if (begin < 0) begin = 0;
		if (end >= indexs.size()) end = indexs.size() - 1;
		int idx = indexs[i];
		for (int j = begin; j <= end; j++)
		{
			if (j == i) continue;
			int j_idx = indexs[j];
			printf("calculating confidence of points %d referencing to %d\n", i, j);
			images[idx].calculateConfidence(images[j_idx], path, camera);
		}
	}
	//for every image
	for (int i = 0; i < indexs.size(); i++)
	{
		int idx = indexs[i];
		printf("adding points of image %d to point cloud\n", i);
		//for each point
		for (int j = 0; j < images[idx].pointcloud.size(); j++)
		{
			if (images[idx].pointcloud[j].low_depth_confidence) continue;
			//for every reference confidence
			double confidence_sum = 0;
			for (int k = 0; k < images[idx].pointcloud[j].confidences.size(); k++)
			{
				confidence_sum += images[idx].pointcloud[j].confidences[k];
			}
			if (images[idx].pointcloud.size() != 0)
			{
				images[idx].pointcloud[j].confidence = confidence_sum / images[idx].pointcloud[j].confidences.size();
			}
			else
			{
				//images[idx].pointcloud[j].confidence = DBL_MAX;
				images[idx].pointcloud[j].confidence = 0;
			}
			//cloud.push_back(images[idx].pointcloud[j]);
		}
		sort(images[idx].pointcloud.begin(), images[idx].pointcloud.end());
		vector<SpacePoint>::iterator it;
		int count;
		for (it = images[idx].pointcloud.begin(), count = 0;it != images[idx].pointcloud.end() && count < images[idx].pointcloud.size()*ratio;it++,count++)
		{
			if (it->low_depth_confidence) continue;
			if (it->low_rgb_confidence) continue;
			cloudofeveryimage.push_back(*it);
		}
	}
	printf("sorting all the points\n");
	//cloud.sort();
	sort(cloudofeveryimage.begin(), cloudofeveryimage.end());
}

void Image::writecloud(string filename)
{
	FILE* file;
	fopen_s(&file, filename.c_str(), "a");
	vector<SpacePoint>::iterator it;
	for (it = pointcloud.begin(); it != pointcloud.end(); ++it)
	{
		if (it->low_depth_confidence) continue;
		fprintf(file, "%lf %lf %lf %d %d %d\n", it->x2, it->y2, it->z2, it->r, it->g, it->b);
	}
	fclose(file);
}

void TestSet::writeRawImages(string filename)
{
	FILE* file;
	fopen_s(&file, (path + filename).c_str(), "w");
	fclose(file);
	for (int i = 0; i < indexs.size(); i++)
	{
		printf("writing image %d\n", i);
		int idx = indexs[i];
		images[idx].writecloud(path + filename);
	}
}

void Image::calculateConfidence(Image &ref, std::string path, Camera camera)
{
	const double step = 0.05;
	const int half_num = 3;
	/*0.85 - 1.15*/
	Mat ref_T_r;
	invert(ref.T, ref_T_r);
	Mat refimg = imread((path + ref.RGB_PATH).c_str());
	for (int i = 0; i < pointcloud.size(); i++)
	{
		if (pointcloud[i].low_depth_confidence) continue;
		double zoom = 1.0;
		int minloss = INT_MAX;
		int curloss = 0;
		for (int j = -half_num; j <= half_num; j++)
		{
			if(j<=0)
				zoom = 1.0 + step * j;
			else
				zoom = 1.0 + step * j * 10;
			double x1 = pointcloud[i].x1 * zoom;
			double y1 = pointcloud[i].y1 * zoom;
			double z1 = pointcloud[i].z1 * zoom;
			Mat XYZ1(4, 1, CV_64FC1);
			XYZ1.at<double>(0, 0) = x1;
			XYZ1.at<double>(1, 0) = y1;
			XYZ1.at<double>(2, 0) = z1;
			XYZ1.at<double>(3, 0) = 1;
			Mat XYZ2 = ref_T_r * T * XYZ1;
			double x1_ref = XYZ2.at<double>(0, 0);
			double y1_ref = XYZ2.at<double>(1, 0);
			double z1_ref = XYZ2.at<double>(2, 0);
			int i_ref = y1_ref * camera.fy_rgb / z1_ref + camera.cy_rgb;
			int j_ref = x1_ref * camera.fx_rgb / z1_ref + camera.cx_rgb;
			if (i_ref < 0 || i_ref >= refimg.rows || j_ref < 0 || j_ref >= refimg.cols)
			{
				continue;
			}
			Vec3b rgb_ref = refimg.at<Vec3b>(i_ref, j_ref);
			uchar r_ref = rgb_ref[2];
			uchar g_ref = rgb_ref[1];
			uchar b_ref = rgb_ref[0];
			int loss = (pointcloud[i].r - r_ref)*(pointcloud[i].r - r_ref)
				+ (pointcloud[i].g - g_ref)*(pointcloud[i].g - g_ref)
				+ (pointcloud[i].b - b_ref)*(pointcloud[i].b - b_ref);
			if (j == 0)
			{
				curloss = loss;
				//if (loss > 1000) pointcloud[i].low_rgb_confidence = true;
			}
			else
			{
				if (loss < minloss)
				{
					minloss = loss;
				}
			}
		}
		double confidence;
		if (curloss == 0)
		{
			confidence = DBL_MAX;
		}
		else
		{
			confidence = (double)minloss / curloss;
		}
		pointcloud[i].confidences.push_back(confidence);
	}
	
}

//void TestSet::writeCloud(std::string filename, double ratio)
//{
//	const double MIN_DIS = 0.001;
//	if (ratio <= 0) ratio = 0.01;
//	if (ratio > 1) ratio = 1;
//	FILE* file;
//	fopen_s(&file, (path + filename).c_str(), "w");
//	vector<SpacePoint *> printed;
//	list<SpacePoint>::iterator it;
//	int count = 0;
//	for (it = cloud.end(), count = 0; it != cloud.begin() && count<cloud.size()*ratio; it--, count++)
//	{
//		if (count % 1000 == 0)
//		{
//			printf("have written %d points to file\n", count);
//		}
//		bool ok = 1;
//		for (int k = 0; k < printed.size(); k++)
//		{
//			if (fabs(printed[k]->x2 - it->x2) < MIN_DIS && fabs(printed[k]->y2 - it->y2) < MIN_DIS && fabs(printed[k]->z2 - it->z2) < MIN_DIS)
//			{
//				ok = 0;
//				break;
//			}
//		}
//
//		if (ok)
//		{
//			fprintf(file, "%lf %lf %lf %d %d %d\n", it->x2, it->y2, it->z2, it->r, it->g, it->b);
//			printed.push_back(&*it);
//		}
//	}
//	fclose(file);
//}

//void TestSet::writeCloudEveryImage(std::string filename)
//{
//	const double MIN_DIS = 0.001;
//	FILE* file;
//	fopen_s(&file, (path + filename).c_str(), "w");
//	vector<SpacePoint *> printed;
//	vector<SpacePoint>::iterator it;
//	int count = 0;
//	for (it = cloudofeveryimage.begin(); it != cloudofeveryimage.end(); it++)
//	{
//		bool ok = 1;
//		for (int k = 0; k < printed.size(); k++)
//		{
//			if (fabs(printed[k]->x2 - it->x2) < MIN_DIS && fabs(printed[k]->y2 - it->y2) < MIN_DIS && fabs(printed[k]->z2 - it->z2) < MIN_DIS)
//			{
//				ok = 0;
//				break;
//			}
//		}
//
//		if (ok)
//		{
//			if (count % 1000 == 0 && count != 0)
//			{
//				printf("have written %d points to file\n", count);
//			}
//			fprintf(file, "%lf %lf %lf %d %d %d\n", it->x2, it->y2, it->z2, it->r, it->g, it->b);
//			printed.push_back(&*it);
//			count++;
//		}
//	}
//	fclose(file);
//}

//void TestSet::writeCloudEveryImagePLY(std::string filename)
//{
//	const double MIN_DIS = 0.001;
//	
//	vector<SpacePoint *> printed;
//	vector<SpacePoint>::iterator it;
//	int count = 0;
//	vector<bool> to_print;
//	for (it = cloudofeveryimage.end(); it != cloudofeveryimage.begin(); it--)
//	{
//		bool ok = 1;
//		/*for (int k = 0; k < printed.size(); k++)
//		{
//			if (fabs(printed[k]->x2 - it->x2) < MIN_DIS && fabs(printed[k]->y2 - it->y2) < MIN_DIS && fabs(printed[k]->z2 - it->z2) < MIN_DIS)
//			{
//				ok = 0;
//				break;
//			}
//		}*/
//
//		if (ok)
//		{
//			if (count % 1000 == 0 && count != 0)
//			{
//				printf("checking %d points to file\n", count);
//			}
//			printed.push_back(&*it);
//			count++;
//			to_print.push_back(true);
//		}
//		else
//		{
//			to_print.push_back(false);
//		}
//	}
//
//	printf("writing file...\n");
//	FILE* file;
//	fopen_s(&file, (path + filename).c_str(), "w");
//	fprintf(file, "ply\n");
//	fprintf(file, "format ascii 1.0\n");
//	/*fprintf(file, "comment author: LBH XSZ\n");
//	fprintf(file, "comment object: untitled\n");*/
//	fprintf(file, "element vertex %d\n",count);
//	fprintf(file, "property float x\n");
//	fprintf(file, "property float y\n");
//	fprintf(file, "property float z\n");
//	fprintf(file, "property uchar red\n");
//	fprintf(file, "property uchar green\n");
//	fprintf(file, "property uchar blue\n");
//	fprintf(file, "end_header\n");
//	for (int i = cloudofeveryimage.size()-1; i >=0; i--)
//	{
//		if (to_print[i])
//		{
//			fprintf(file, "%lf %lf %lf %d %d %d\n", cloudofeveryimage[i].x2, cloudofeveryimage[i].y2, cloudofeveryimage[i].z2, cloudofeveryimage[i].r, cloudofeveryimage[i].g, cloudofeveryimage[i].b);
//		}
//	}
//	fprintf(file, "\n");
//	fclose(file);
//}


void Image::denoiseAccordingToDepth(Image &ref, std::string path, Camera camera)
{
	static int larger_than;
	static int total;
	Mat ref_T_r;
	invert(ref.T, ref_T_r);
	Mat refdepth = imread((path + ref.DEPTH_PATH).c_str());
	vector<SpacePoint>::iterator it;
	for (it = pointcloud.begin(); it != pointcloud.end(); it++)
	{
		double x2 = it->x2;
		double y2 = it->y2;
		double z2 = it->z2;
		Mat XYZ1(4, 1, CV_64FC1);
		XYZ1.at<double>(0, 0) = x2;
		XYZ1.at<double>(1, 0) = y2;
		XYZ1.at<double>(2, 0) = z2;
		XYZ1.at<double>(3, 0) = 1;
		Mat XYZ2 = ref_T_r * XYZ1;
		
		double x1_ref = XYZ2.at<double>(0, 0);
		double y1_ref = XYZ2.at<double>(1, 0);
		double z1_ref = XYZ2.at<double>(2, 0);
		int i_ref = y1_ref * camera.fy_rgb / z1_ref + camera.cy_rgb;
		int j_ref = x1_ref * camera.fx_rgb / z1_ref + camera.cx_rgb;
		

		if (i_ref < 0 || i_ref >= refdepth.rows || j_ref < 0 || j_ref >= refdepth.cols)
		{
			continue;
		}

		ushort depth = refdepth.at<ushort>(i_ref, j_ref);
		double z = (double)depth / camera.camera_factor;

		/*if (it->i > 220 && it->i < 260 && it->j>300 && it->j < 340)
		{
			cout << XYZ1 << endl;
			cout << XYZ2 << endl;
			cout << it->i << " " << it->j << " " << it->d << endl;
			cout << i_ref << " " << j_ref << endl;
			printf("%lf %lf\n", z, z1_ref);
			printf("\n");
		}*/
		
		//if (z < z1_ref*0.90 || z > z1_ref*1.10)
		if (z < z1_ref*0.50 || z > z1_ref*1.50)
		{
			it->low_depth_confidence = true;
		}
		else
		{
			count_false++;
		}

		//it->low_depth_confidence = (z < z1_ref*0.5 || z > z1_ref*1.5);
		//it->low_depth_confidence = true;
	}
}

void TestSet::denoiseAccordingToDepth(int range)
{
	for (int i = 0; i < indexs.size(); i++)
	{
		int begin = i - range;
		int end = i + range;
		if (begin < 0) begin = 0;
		if (end >= indexs.size()) end = indexs.size() - 1;
		int idx = indexs[i];
		for (int j = begin; j <= end; j++)
		{
			if (j == i) continue;
			int j_idx = indexs[j];
			printf("denoise points %d referencing to %d\n", i, j);
			images[idx].denoiseAccordingToDepth(images[j_idx], path, camera);
		}
		//printf("%d\n",count_false);
	}
}

void TestSet::removeduplicate()
{
	/*vector<SpacePoint> cloudofeveryimage;*/
	vector<SpacePoint>::iterator it;
	for (it = cloudofeveryimage.begin(); it != cloudofeveryimage.end(); it++)
	{
		Vec3i xyz;
		xyz[0] = 500*it->x2;
		xyz[1] = 500*it->y2;
		xyz[2] = 500*it->z2;
		hashcloud[xyz] = &*it;
	}
}

void TestSet::writeHashCloud(std::string filename)
{
	FILE* file;
	fopen_s(&file, (path + filename).c_str(), "w");
	unordered_map<cv::Vec3i, SpacePoint *>::iterator it;
	for (it = hashcloud.begin(); it != hashcloud.end(); it++)
	{
		fprintf(file, "%lf %lf %lf %d %d %d\n", it->second->x2, it->second->y2, it->second->z2, it->second->r, it->second->g, it->second->b);
	}
	fclose(file);
}

void TestSet::generateNormal()
{

	Kdtree::KdNodeVector nodes;
	unordered_map<cv::Vec3i, SpacePoint*>::iterator it;
	for (it=hashcloud.begin();it!=hashcloud.end();it++)
	{
		vector<double> point(3);
		point[0] = it->second->x2;
		point[1] = it->second->y2;
		point[2] = it->second->z2;
		nodes.push_back(point);
	}
	Kdtree::KdTree tree(&nodes);

	for (it = hashcloud.begin(); it != hashcloud.end(); it++)
	{
		vector<double> point(3);
		point[0] = it->second->x2;
		point[1] = it->second->y2;
		point[2] = it->second->z2;
		Kdtree::KdNodeVector result;
		tree.k_nearest_neighbors(point, 10, &result);
		double x_sum = point[0];
		double y_sum = point[1];
		double z_sum = point[2];
		for (int j = 0; j < result.size(); j++)
		{
			x_sum += result[j].point[0];
			y_sum += result[j].point[1];
			z_sum += result[j].point[2];
		}
		double x_avg = x_sum / (result.size() + 1);
		double y_avg = y_sum / (result.size() + 1);
		double z_avg = z_sum / (result.size() + 1);
		Mat A = Mat::zeros(3, 3, CV_64FC1);
		for (int j = 0; j < result.size(); j++)
		{
			A.at<double>(0, 0) += (result[j].point[0] - x_avg)*(result[j].point[0] - x_avg);
			A.at<double>(0, 1) += (result[j].point[0] - x_avg)*(result[j].point[1] - y_avg);
			A.at<double>(0, 2) += (result[j].point[0] - x_avg)*(result[j].point[2] - z_avg);
			A.at<double>(1, 0) += (result[j].point[1] - y_avg)*(result[j].point[0] - x_avg);
			A.at<double>(1, 1) += (result[j].point[1] - y_avg)*(result[j].point[1] - y_avg);
			A.at<double>(1, 2) += (result[j].point[1] - y_avg)*(result[j].point[2] - z_avg);
			A.at<double>(2, 0) += (result[j].point[2] - z_avg)*(result[j].point[0] - x_avg);
			A.at<double>(2, 1) += (result[j].point[2] - z_avg)*(result[j].point[1] - y_avg);
			A.at<double>(2, 2) += (result[j].point[2] - z_avg)*(result[j].point[2] - z_avg);
		}
		cv::Mat eValuesMat;
		cv::Mat eVectorsMat;
		eigen(A, eValuesMat, eVectorsMat);
		int idx_min;
		double eValue_min = DBL_MAX;
		for (int k = 0; k < 3; k++)
		{
			if (eValuesMat.at<double>(k, 0) <= eValue_min)
			{
				eValue_min = eValuesMat.at<double>(k, 0);
				idx_min = k;
			}
		}
		it->second->nx = eVectorsMat.at<double>(idx_min, 0);
		it->second->ny = eVectorsMat.at<double>(idx_min, 1);
		it->second->nz = eVectorsMat.at<double>(idx_min, 2);
	}
	
}

void TestSet::writeNormal(std::string filename)
{
	FILE* file;
	fopen_s(&file, (path + filename).c_str(), "w");
	unordered_map<cv::Vec3i, SpacePoint *>::iterator it;
	for (it = hashcloud.begin(); it != hashcloud.end(); it++)
	{
		double va_x = it->second->tx - it->second->x2;
		double va_y = it->second->ty - it->second->y2;
		double va_z = it->second->tz - it->second->z2;
		if ((va_x*it->second->nx + va_y * it->second->ny + va_z * it->second->nz) > 0)
		{
			fprintf(file, "%lf %lf %lf %lf %lf %lf\n",
				it->second->x2,
				it->second->y2,
				it->second->z2,
				it->second->nx,
				it->second->ny,
				it->second->nz
			);
		}
		else
		{
			fprintf(file, "%lf %lf %lf %lf %lf %lf\n",
				it->second->x2,
				it->second->y2,
				it->second->z2,
				-it->second->nx,
				-it->second->ny,
				-it->second->nz
			);
		}

	}
	fclose(file);
}