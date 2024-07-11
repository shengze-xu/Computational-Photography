#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "DepthRecover.h"
#include "GCoptimization.h"

using namespace std;
using namespace cv;

extern FILE *drlog;

//GCoptimizationGridGraph *GC;
GCoptimizationGeneralGraph *GC;

double square(double x)
{
	return x * x;
}

//Vec3d depthToPoint(int i, int j, double d, Mat P, double cx, double cy, double fx, double fy)
//{
//	double z = 1 / d;
//	double x = (j - cx)*z / fx;
//	double y = (i - cy)*z / fy;
//	Mat point_before_transform(4, 1, CV_64FC1);
//	point_before_transform.at<double>(0, 0) = x;
//	point_before_transform.at<double>(1, 0) = y;
//	point_before_transform.at<double>(2, 0) = z;
//	point_before_transform.at<double>(3, 0) = 1;
//	Mat point_after_transform = P * point_before_transform;
//	Vec3d res;
//	res[0] = point_after_transform.at<double>(0, 0);
//	res[1] = point_after_transform.at<double>(1, 0);
//	res[2] = point_after_transform.at<double>(2, 0);
//	return res;
//}

//Vec3d pointToDepth(Vec3d xyz, Mat P, double cx, double cy, double fx, double fy)
//{
//	//cout << P << endl;
//	Mat P_r;
//	invert(P, P_r);
//	Mat XYZ2(4, 1, CV_64FC1);
//	XYZ2.at<double>(0, 0) = xyz[0];
//	XYZ2.at<double>(1, 0) = xyz[1];
//	XYZ2.at<double>(2, 0) = xyz[2];
//	XYZ2.at<double>(3, 0) = 1;
//	Mat XYZ1 = P_r * XYZ2;
//	double x1 = XYZ1.at<double>(0, 0);
//	double y1 = XYZ1.at<double>(1, 0);
//	double z1 = XYZ1.at<double>(2, 0);
//	int i_img = y1 * fy / z1 + cy;
//	int j_img = x1 * fx / z1 + cx;
//	double d = 1 / z1;
//	Vec3d ret;
//	ret[0] = i_img;
//	ret[1] = j_img;
//	ret[2] = d;
//	return ret;
//}

void DRPicture::generateP()
{
	P = Mat(4, 4, CV_64FC1);
	P.at<double>(0, 0) = 1 - 2 * QY * QY - 2 * QZ * QZ;
	P.at<double>(0, 1) = 2 * QX * QY - 2 * QW * QZ;
	P.at<double>(0, 2) = 2 * QX * QZ + 2 * QW * QY;
	P.at<double>(1, 0) = 2 * QX * QY + 2 * QW * QZ;
	P.at<double>(1, 1) = 1 - 2 * QX * QX - 2 * QZ * QZ;
	P.at<double>(1, 2) = 2 * QY * QZ - 2 * QW * QX;
	P.at<double>(2, 0) = 2 * QX * QZ - 2 * QW * QY;
	P.at<double>(2, 1) = 2 * QY * QZ + 2 * QW * QX;
	P.at<double>(2, 2) = 1 - 2 * QX * QX - 2 * QY * QY;

	P.at<double>(0, 3) = TX;
	P.at<double>(1, 3) = TY;
	P.at<double>(2, 3) = TZ;

	P.at<double>(3, 0) = 0;
	P.at<double>(3, 1) = 0;
	P.at<double>(3, 2) = 0;
	P.at<double>(3, 3) = 1;
}

void DRPicture::generateR()
{
	R = Mat(3, 3, CV_64FC1);
	R.at<double>(0, 0) = 1 - 2 * QY * QY - 2 * QZ * QZ;
	R.at<double>(0, 1) = 2 * QX * QY - 2 * QW * QZ;
	R.at<double>(0, 2) = 2 * QX * QZ + 2 * QW * QY;
	R.at<double>(1, 0) = 2 * QX * QY + 2 * QW * QZ;
	R.at<double>(1, 1) = 1 - 2 * QX * QX - 2 * QZ * QZ;
	R.at<double>(1, 2) = 2 * QY * QZ - 2 * QW * QX;
	R.at<double>(2, 0) = 2 * QX * QZ - 2 * QW * QY;
	R.at<double>(2, 1) = 2 * QY * QZ + 2 * QW * QX;
	R.at<double>(2, 2) = 1 - 2 * QX * QX - 2 * QY * QY;
	
	R_T = R.t();
}

void DRPicture::generateT()
{
	T = Mat(3, 1, CV_64FC1);
	T.at<double>(0, 0) = TX;
	T.at<double>(1, 0) = TY;
	T.at<double>(2, 0) = TZ;
}

void DRPicture::generateK()
{
	K = Mat::zeros(3, 3, CV_64FC1);
	K.at<double>(0, 0) = fx_rgb;
	K.at<double>(0, 2) = cx_rgb;
	K.at<double>(1, 1) = fy_rgb;
	K.at<double>(1, 2) = cy_rgb;
	K.at<double>(2, 2) = 1;

	invert(K, K_1);
}

void DepthRecoverSolution::loadInfo(std::string p)
{
	const int FROM_FRONT_XX_IMAGES = 2;
	const int STEP = 1;

	FILE* file;
	double omit;
	char buffer[100];
	DRPicture pic;
	path = p;

	fopen_s(&file, (path + "info/camera.txt").c_str(), "r");
	if (file == NULL) printf("error\n");
	fscanf_s(file, "%lf", &fx_rgb);
	fscanf_s(file, "%lf", &fy_rgb);
	fscanf_s(file, "%lf", &cx_rgb);
	fscanf_s(file, "%lf", &cy_rgb);
	pic.cx_rgb = cx_rgb;
	pic.cy_rgb = cy_rgb;
	pic.fx_rgb = fx_rgb;
	pic.fy_rgb = fy_rgb;
	fclose(file);

	fopen_s(&file, (path + "info/associate_with_groundtruth.txt").c_str(), "r");
	while (fscanf_s(file, "%lf", &omit) > 0)
	{
		fscanf_s(file, "%s", buffer, 100);
		pic.rgb_path = buffer;
		fscanf_s(file, "%lf", &omit);
		fscanf_s(file, "%s", buffer, 100);
		fscanf_s(file, "%lf", &omit);
		fscanf_s(file, "%lf", &pic.TX);
		fscanf_s(file, "%lf", &pic.TY);
		fscanf_s(file, "%lf", &pic.TZ);
		fscanf_s(file, "%lf", &pic.QX);
		fscanf_s(file, "%lf", &pic.QY);
		fscanf_s(file, "%lf", &pic.QZ);
		fscanf_s(file, "%lf", &pic.QW);
		pictures.push_back(pic);
	}
	fclose(file);

	int from_front = 0;
	if (pictures.size() < FROM_FRONT_XX_IMAGES) from_front = pictures.size();
	else from_front = FROM_FRONT_XX_IMAGES;
	for (int i = 0; i < from_front; i += STEP)
	{
		fprintf(drlog, "index:%d\n",i);
		indexs.push_back(i);
	}
}

void DepthRecoverSolution::loadRGBImgAndCreateDisparity()
{
	for (int i = 0; i < indexs.size(); i++)
	{
		int idx = indexs[i];
		pictures[idx].rgb = imread((path+ pictures[idx].rgb_path).c_str());
		pictures[idx].disparity = Mat(pictures[idx].rgb.size(), CV_16UC1);
		pictures[idx].generateP();
		pictures[idx].generateR();
		pictures[idx].generateT();
		pictures[idx].generateK();
	}
}

struct ForDataFn
{
	std::vector<DRPicture>* pictures;
	std::vector<int>* indexs;
};

//return a Mat 3*1 (i,j,1)
Mat convert_xt_to_xtp(Mat xt, Mat K_1_src, Mat R_src, Mat R_T_dst, 
	Mat K_dst, double dxt, Mat T_src, Mat T_dst)
{
	/*printf("xt:\n");
	cout << xt << endl;
	printf("K_1_src:\n");
	cout << K_1_src << endl;
	printf("R_src:\n");
	cout << R_src << endl;
	printf("R_T_dst:\n");
	cout << R_T_dst << endl;
	printf("K_dst:\n");
	cout << K_dst << endl;*/

	return K_dst * R_T_dst*R_src*K_1_src*xt + dxt * K_dst*R_T_dst*(T_src - T_dst);
	///*printf("label1\n");*/
	//Mat part1 = K_dst * R_T_dst * R_src * K_1_src * xt;
	////cout << part1 << endl;
	////printf("label2\n");
	//Mat part2 = K_dst * R_T_dst*(T_src - T_dst);
	///*cout << part2 << endl;
	//printf("label3\n");*/
	//Mat ret = part1 + dxt * part2;
	////cout << ret << endl;
	//return ret;
}

const double D_min = 0.1;
const double D_max = 10.0;
const double MAX_K = 10.0;
const int t_range = 1;
const double SIGMA_C = 10;
const double SIGMA_D = 3;
const double EPSILON = 0.1;
const double WS = 0.5;
const double YITA = 4;


//int dataFn_init(int p, int l, void *data)
//{
//	//const
//	ForDataFn *myData = (ForDataFn *)data;
//	int idx0 = (*(myData->indexs))[0];
//	int rows = (*(myData->pictures))[idx0].rgb.rows;
//	int cols = (*(myData->pictures))[idx0].rgb.cols;
//	int img_num = myData->indexs->size();
//	//p related 
//	int sn = p / (rows*cols);
//	int idx = (*(myData->indexs))[sn];
//	int i_img = (p % (rows*cols)) / cols;
//	int j_img = (p % (rows*cols)) % cols;
//	double cx = (*(myData->pictures))[idx].cx_rgb;
//	double cy = (*(myData->pictures))[idx].cy_rgb;
//	double fx = (*(myData->pictures))[idx].fx_rgb;
//	double fy = (*(myData->pictures))[idx].fy_rgb;
//	
//	double maxLinit = 0;
//	for (int k = 0; k <= MAX_K; k++)
//	{
//		double d_k = (MAX_K - k) / MAX_K * D_min + k / MAX_K * D_max;
//
//		double Linit_xt_dk = 0;
//		for (int i = idx - t_range; i <= idx + t_range; i++)
//		{
//			if (i < 0) continue;
//			if (i >= img_num) continue;
//			if (i == idx) continue;
//			int idx_i = (*(myData->indexs))[i];
//			Vec3b bgr = (*(myData->pictures))[idx].rgb.at<Vec3b>(i_img, j_img);
//			double r = bgr[2];
//			//printf("%lf", r);
//			double g = bgr[1];
//			double b = bgr[0];
//			Vec3d xyz = depthToPoint(i_img, j_img, d_k, (*(myData->pictures))[idx].P, cx, cy, fx, fy);
//			Vec3d ijd = pointToDepth(xyz, (*(myData->pictures))[idx_i].P, cx, cy, fx, fy);
//			int i_ref = ijd[0];
//			int j_ref = ijd[1];
//			if (i_ref < 0 || i_ref >= rows || j_ref < 0 || j_ref >= cols) continue;
//			Vec3b bgr_ref = (*(myData->pictures))[idx_i].rgb.at<Vec3b>(i_ref, j_ref);
//			double rf = bgr_ref[2];
//			//printf("%lf", rf);
//			double gf = bgr_ref[1];
//			double bf = bgr_ref[0];
//
//			/*double r_rf = r - rf;
//			double g_gf = g - gf;
//			double b_bf = b - bf;
//			double square_r_rf = square(r_rf);
//			double square_g_gf = square(g_gf);
//			double square_b_bf = square(b_bf);*/
//			double dis = sqrt(square(r - rf) + square(g - gf) + square(b - bf));
//			//double dis = square_r_rf + square_g_gf + square_b_bf;
//			double pc = SIGMA_C / (SIGMA_C + dis);
//			Linit_xt_dk += pc;
//		}
//		if (Linit_xt_dk > maxLinit)
//		{
//			maxLinit = Linit_xt_dk;
//		}
//	}
//	double u = 1.0 / maxLinit;
//
//	double Linit = 0;
//	double dxt = (MAX_K - l) / MAX_K * D_min + l / MAX_K * D_max;
//	for (int i = idx - t_range; i <= idx + t_range; i++)
//	{
//		if (i < 0) continue;
//		if (i >= img_num) continue;
//		if (i == idx) continue;
//		int idx_i = (*(myData->indexs))[i];
//		Vec3b bgr = (*(myData->pictures))[idx].rgb.at<Vec3b>(i_img, j_img);
//		double r = bgr[2];
//		double g = bgr[1];
//		double b = bgr[0];
//		Vec3d xyz = depthToPoint(i_img, j_img, dxt, (*(myData->pictures))[idx].P, cx, cy, fx, fy);
//		Vec3d ijd = pointToDepth(xyz, (*(myData->pictures))[idx_i].P, cx, cy, fx, fy);
//		int i_ref = ijd[0];
//		int j_ref = ijd[1];
//		if (i_ref < 0 || i_ref >= rows || j_ref < 0 || j_ref >= cols) continue;
//		Vec3b bgr_ref = (*(myData->pictures))[idx_i].rgb.at<Vec3b>(i_ref, j_ref);
//		double rf = bgr_ref[2];
//		double gf = bgr_ref[1];
//		double bf = bgr_ref[0];
//
//		double dis = square(r - rf) + square(g - gf) + square(b - bf);
//		double pc = SIGMA_C / (SIGMA_C + dis);
//		Linit += pc;
//	}
//	
//
//	double E = 1 - u * Linit;
//	return E;
//}

double smoothFn(int p1, int p2, int l1, int l2, void *data)
{
	ForDataFn *myData = (ForDataFn *)data;
	//get const
	int idx0 = (*(myData->indexs))[0];
	int rows = (*(myData->pictures))[idx0].rgb.rows;
	int cols = (*(myData->pictures))[idx0].rgb.cols;
	int img_num = myData->indexs->size();
	//get point1 related
	int sn1 = p1 / (rows*cols);
	int idx1 = (*(myData->indexs))[sn1];
	int i1_img = (p1 % (rows*cols)) / cols;
	int j1_img = (p1 % (rows*cols)) % cols;
	//get point2 related
	int sn2 = p2 / (rows*cols);
	int idx2 = (*(myData->indexs))[sn2];
	int i2_img = (p2 % (rows*cols)) / cols;
	int j2_img = (p2 % (rows*cols)) % cols;

	double dxt = (MAX_K - l1) / MAX_K * D_min + l1 / MAX_K * D_max;
	double dyt = (MAX_K - l2) / MAX_K * D_min + l2 / MAX_K * D_max;

	if (idx1 != idx2) return 0;
	//if (abs(i1_img - i2_img) > 2 || abs(j1_img - j2_img) > 2) return 0;

	
	Vec3b bgrxt = (*(myData->pictures))[idx1].rgb.at<Vec3b>(i1_img, j1_img);
	double bxt = bgrxt[0];
	double gxt = bgrxt[1];
	double rxt = bgrxt[2];

	int nxt_count = 0;
	double sum_1 = 0;
	int i_xn, j_xn;
	//up
	i_xn = i1_img - 1;
	j_xn = j1_img;
	if (i_xn >= 0)
	{
		nxt_count++;
		Vec3b bgrypt = (*(myData->pictures))[idx1].rgb.at<Vec3b>(i_xn, j_xn);
		double bypt = bgrypt[0];
		double gypt = bgrypt[1];
		double rypt = bgrypt[2];
		double dis_ixy = sqrt(square(bxt - bypt) + square(gxt - gypt) + square(rxt - rypt));
		sum_1 += 1 / (dis_ixy + EPSILON);
	}
	//left
	i_xn = i1_img;
	j_xn = j1_img - 1;
	if (j_xn >= 0)
	{
		nxt_count++;
		Vec3b bgrypt = (*(myData->pictures))[idx1].rgb.at<Vec3b>(i_xn, j_xn);
		double bypt = bgrypt[0];
		double gypt = bgrypt[1];
		double rypt = bgrypt[2];
		double dis_ixy = sqrt(square(bxt - bypt) + square(gxt - gypt) + square(rxt - rypt));
		sum_1 += 1 / (dis_ixy + EPSILON);
	}
	//down
	i_xn = i1_img + 1;
	j_xn = j1_img;
	if (i_xn < rows)
	{
		nxt_count++;
		Vec3b bgrypt = (*(myData->pictures))[idx1].rgb.at<Vec3b>(i_xn, j_xn);
		double bypt = bgrypt[0];
		double gypt = bgrypt[1];
		double rypt = bgrypt[2];
		double dis_ixy = sqrt(square(bxt - bypt) + square(gxt - gypt) + square(rxt - rypt));
		sum_1 += 1 / (dis_ixy + EPSILON);
	}
	//right
	i_xn = i1_img;
	j_xn = j1_img + 1;
	if (j_xn < cols)
	{
		nxt_count++;
		Vec3b bgrypt = (*(myData->pictures))[idx1].rgb.at<Vec3b>(i_xn, j_xn);
		double bypt = bgrypt[0];
		double gypt = bgrypt[1];
		double rypt = bgrypt[2];
		double dis_ixy = sqrt(square(bxt - bypt) + square(gxt - gypt) + square(rxt - rypt));
		sum_1 += 1 / (dis_ixy + EPSILON);
	}


	double u_lamda_xt = nxt_count / sum_1;
	Vec3b bgryt = (*(myData->pictures))[idx1].rgb.at<Vec3b>(i2_img, j2_img);
	double byt = bgryt[0];
	double gyt = bgryt[1];
	double ryt = bgryt[2];
	double disxy = sqrt(square(bxt - byt) + square(gxt - gyt) + square(rxt - ryt));
	double lamada_xt_yt = WS * u_lamda_xt / (disxy + EPSILON);

	double rou_dxt_dyt;
	if (fabs(dxt - dyt) < YITA) rou_dxt_dyt = fabs(dxt - dyt);
	else rou_dxt_dyt = YITA;
	double Es = lamada_xt_yt * rou_dxt_dyt;
	//fprintf(drlog, "[Es]:p1=%d,l1=%d,p2=%d,l2=%d,lamada_xt_yt=%lf,rou_dxt_dyt=%lf,Es=%lf\n", p1, l1, p2, l2, lamada_xt_yt, rou_dxt_dyt, Es);
	return Es;
}

//maybe useless
//double getmaxLxtdt(int* result, ForDataFn *data)
//{
//
//	ForDataFn *myData = (ForDataFn *)data;
//	int idx0 = (*(myData->indexs))[0];
//	double cx = (*(myData->pictures))[idx0].cx_rgb;
//	double cy = (*(myData->pictures))[idx0].cy_rgb;
//	double fx = (*(myData->pictures))[idx0].fx_rgb;
//	double fy = (*(myData->pictures))[idx0].fy_rgb;
//	int rows = (*(myData->pictures))[idx0].rgb.rows;
//	int cols = (*(myData->pictures))[idx0].rgb.cols;
//	int img_num = myData->indexs->size();
//	
//	double maxLxtdt = 0;
//	for (int p = 0; p < rows*cols*img_num;p++)
//	{
//		int l = result[p];
//		int idx = p / (rows*cols);
//		int i_img = (p % (rows*cols)) / cols;
//		int j_img = (p % (rows*cols)) % cols;
//		double d = (MAX_K - l) / MAX_K * D_min + l / MAX_K * D_max;
//		double L_xt_dk = 0;
//		for (int i = idx - t_range; i <= idx + t_range; i++)
//		{
//			if (i < 0) continue;
//			if (i >= img_num) continue;
//			if (i == idx) continue;
//			int idx_i = (*(myData->indexs))[i];
//			Vec3b bgr = (*(myData->pictures))[idx].rgb.at<Vec3b>(i_img, j_img);
//			double r = bgr[2];
//			double g = bgr[1];
//			double b = bgr[0];
//			Vec3d xyz = depthToPoint(i_img, j_img, d, (*(myData->pictures))[idx].P, cx, cy, fx, fy);
//			Vec3d ijd = pointToDepth(xyz, (*(myData->pictures))[idx_i].P, cx, cy, fx, fy);
//			int i_ref = ijd[0];
//			int j_ref = ijd[1];
//			if (i_ref < 0 || i_ref >= rows || j_ref < 0 || j_ref >= cols) continue;
//			Vec3b bgr_ref = (*(myData->pictures))[idx_i].rgb.at<Vec3b>(i_ref, j_ref);
//			double rf = bgr_ref[2];
//			double gf = bgr_ref[1];
//			double bf = bgr_ref[0];
//
//			double dis = sqrt(square(r - rf) + square(g - gf) + square(b - bf));
//			double pc = SIGMA_C / (SIGMA_C + dis);
//			double maxexp = 0;
//			for (int m = i_ref - 2; m <= i_ref + 2; m++)
//			{
//				for (int n = j_ref - 2; n <= j_ref + 2; n++)
//				{
//					if (m < 0 || m >= rows || n < 0 || n >= cols) continue;
//					if (m == i_ref && n == j_ref) continue;
//					double d = (MAX_K - 5) / MAX_K * D_min + 5 / MAX_K * D_max;
//					Vec3d xyz = depthToPoint(m, n, d, (*(myData->pictures))[i].P, cx, cy, fx, fy);
//					Vec3d ijd = pointToDepth(xyz, (*(myData->pictures))[idx].P, cx, cy, fx, fy);
//					int i2 = ijd[0];
//					int j2 = ijd[1];
//					double point_dis2 = square(i2 - i_img) + square(j2 - j_img);
//					double expinpv = exp(-point_dis2 / 2 / square(SIGMA_D));
//					if (expinpv > maxexp) maxexp = expinpv;
//				}
//			}
//			double pv = maxexp;
//			L_xt_dk += pc * pc;
//		}
//		if (L_xt_dk > maxLxtdt)
//		{
//			maxLxtdt = L_xt_dk;
//		}
//	}
//	return maxLxtdt;
//}

double Lxtdxt_init(int sn, int i_img, int j_img, double dxt, ForDataFn *myData)
{
	int idx0 = (*(myData->indexs))[0];
	int rows = (*(myData->pictures))[idx0].rgb.rows;
	int cols = (*(myData->pictures))[idx0].rgb.cols;
	int img_num = myData->indexs->size();
	int idx = (*(myData->indexs))[sn];
	double cx = (*(myData->pictures))[idx].cx_rgb;
	double cy = (*(myData->pictures))[idx].cy_rgb;
	double fx = (*(myData->pictures))[idx].fx_rgb;
	double fy = (*(myData->pictures))[idx].fy_rgb;

	Vec3b bgr = (*(myData->pictures))[idx].rgb.at<Vec3b>(i_img, j_img);
	double r = bgr[2];
	double g = bgr[1];
	double b = bgr[0];

	Mat xt(3, 1, CV_64FC1);
	xt.at<double>(0, 0) = j_img;
	xt.at<double>(1, 0) = i_img;
	xt.at<double>(2, 0) = 1;

	double L = 0;
	for (int i = sn - t_range; i <= sn + t_range; i++)
	{
		if (i < 0) continue;
		if (i >= img_num) continue;
		if (i == sn) continue;
		int ref_idx = (*(myData->indexs))[i];

		Mat xtp = convert_xt_to_xtp(xt, (*(myData->pictures))[idx].K_1, (*(myData->pictures))[idx].R,
			(*(myData->pictures))[ref_idx].R_T, (*(myData->pictures))[ref_idx].K, dxt,
			(*(myData->pictures))[idx].T, (*(myData->pictures))[ref_idx].T);
		int i_ref = xtp.at<double>(1, 0);
		int j_ref = xtp.at<double>(0, 0);
		if (i_ref < 0 || i_ref >= rows || j_ref < 0 || j_ref >= cols) continue;
		Vec3b bgr_ref = (*(myData->pictures))[ref_idx].rgb.at<Vec3b>(i_ref, j_ref);
		double rf = bgr_ref[2];
		double gf = bgr_ref[1];
		double bf = bgr_ref[0];
		double disI = square(r - rf) + square(g - gf) + square(b - bf);
		double pc = SIGMA_C / (SIGMA_C + disI);

		L += pc;
	}
	//fprintf(drlog,"L:%lf, %lf\n",dxt, L);
	return L;
}

double dataFn_init(int p, int l, void *data)
{
	ForDataFn *myData = (ForDataFn *)data;
	int idx0 = (*(myData->indexs))[0];
	int rows = (*(myData->pictures))[idx0].rgb.rows;
	int cols = (*(myData->pictures))[idx0].rgb.cols;
	int img_num = myData->indexs->size();
	int sn = p / (rows*cols);
	int i_img = (p % (rows*cols)) / cols;
	int j_img = (p % (rows*cols)) % cols;
	int idx = (*(myData->indexs))[sn];

	double maxL = 0;
	for (int i = 0; i <= MAX_K; i++)
	{
		double d = (MAX_K - i) / MAX_K * D_min + i / MAX_K * D_max;
		//printf("d:%lf\n", d);
		double curL = Lxtdxt_init(sn, i_img, j_img, d, myData);
		if (curL > maxL) maxL = curL;
	}

	double dxt = (MAX_K - l) / MAX_K * D_min + l / MAX_K * D_max;
	double L = Lxtdxt_init(sn, i_img, j_img, dxt, myData);

	if (maxL == 0) return 0;
	double Ed = 1 - L / maxL;
	//fprintf(drlog, "[Ed_init] p=%d,l=%d,Ed=%lf\n", p, l, Ed);
	return Ed;
}

double Lxtdxt(int sn, int i_img, int j_img, double dxt, ForDataFn *myData)
{
	int idx0 = (*(myData->indexs))[0];
	int rows = (*(myData->pictures))[idx0].rgb.rows;
	int cols = (*(myData->pictures))[idx0].rgb.cols;
	int img_num = myData->indexs->size();
	int idx = (*(myData->indexs))[sn];
	double cx = (*(myData->pictures))[idx].cx_rgb;
	double cy = (*(myData->pictures))[idx].cy_rgb;
	double fx = (*(myData->pictures))[idx].fx_rgb;
	double fy = (*(myData->pictures))[idx].fy_rgb;
	
	Vec3b bgr = (*(myData->pictures))[idx].rgb.at<Vec3b>(i_img, j_img);
	double r = bgr[2];
	double g = bgr[1];
	double b = bgr[0];
	
	Mat xt(3, 1, CV_64FC1);
	xt.at<double>(0, 0) = j_img;
	xt.at<double>(1, 0) = i_img;
	xt.at<double>(2, 0) = 1;

	double L = 0;
	for (int i = sn - t_range; i <= sn + t_range; i++)
	{
		if (i < 0) continue;
		if (i >= img_num) continue;
		if (i == sn) continue;
		int ref_idx = (*(myData->indexs))[i];

		Mat xtp = convert_xt_to_xtp(xt, (*(myData->pictures))[idx].K_1, (*(myData->pictures))[idx].R,
			(*(myData->pictures))[ref_idx].R_T, (*(myData->pictures))[ref_idx].K, dxt,
			(*(myData->pictures))[idx].T, (*(myData->pictures))[ref_idx].T);
		int i_ref = xtp.at<double>(1, 0);
		int j_ref = xtp.at<double>(0, 0);
		if (i_ref < 0 || i_ref >= rows || j_ref < 0 || j_ref >= cols) continue;
		Vec3b bgr_ref = (*(myData->pictures))[ref_idx].rgb.at<Vec3b>(i_ref, j_ref);
		double rf = bgr_ref[2];
		double gf = bgr_ref[1];
		double bf = bgr_ref[0];
		double disI = square(r - rf) + square(g - gf) + square(b - bf);
		double pc = SIGMA_C / (SIGMA_C + disI);


		double maxexp = 0;
		for (int m = i_ref - 2; m <= i_ref + 2; m++)
		{
			for (int n = j_ref - 2; n <= j_ref + 2; n++)
			{
				if (m < 0 || m >= rows || n < 0 || n >= cols) continue;
				if (m == i_ref && n == j_ref) continue;
				int p_ref = rows * cols*i + m * cols + n;
				int l_ref = GC->whatLabel(p_ref);
				double d = (MAX_K - l_ref) / MAX_K * D_min + l_ref / MAX_K * D_max;
				Mat xyInW(3, 1, CV_64FC1);
				xyInW.at<double>(0, 0) = n;
				xyInW.at<double>(1, 0) = m;
				xyInW.at<double>(2, 0) = 1;
				Mat xy2 = convert_xt_to_xtp(xyInW, (*(myData->pictures))[ref_idx].K_1, 
					(*(myData->pictures))[ref_idx].R, (*(myData->pictures))[idx].R_T, 
					(*(myData->pictures))[idx].K, d, 
					(*(myData->pictures))[ref_idx].T, (*(myData->pictures))[idx].T);
				int j2 = xy2.at<double>(0, 0);
				int i2 = xy2.at<double>(1, 0);
				
				double point_dis2 = square(i2 - i_img) + square(j2 - j_img);
				double expinpv = exp(-point_dis2 / 2 / square(SIGMA_D));
				if (expinpv > maxexp) maxexp = expinpv;
			}
		}
		double pv = maxexp;
		L += pc * pv;
	}
	return L;
}

double dataFn_iterative(int p, int l, void *data)
{
	ForDataFn *myData = (ForDataFn *)data;
	int idx0 = (*(myData->indexs))[0];
	int rows = (*(myData->pictures))[idx0].rgb.rows;
	int cols = (*(myData->pictures))[idx0].rgb.cols;
	int img_num = myData->indexs->size();
	int sn = p / (rows*cols);
	int i_img = (p % (rows*cols)) / cols;
	int j_img = (p % (rows*cols)) % cols;
	int idx = (*(myData->indexs))[sn];

	double maxL = 0;
	for (int i = 0; i < MAX_K; i++)
	{
		double d = (MAX_K - i) / MAX_K * D_min + i / MAX_K * D_max;
		double curL = Lxtdxt(sn, i_img, j_img, d, myData);
		if (curL > maxL) maxL = curL;
	}

	double dxt = (MAX_K - l) / MAX_K * D_min + l / MAX_K * D_max;
	double L = Lxtdxt(sn, i_img, j_img, dxt, myData);
	if (maxL == 0) return 0;
	double Ed = 1 - L / maxL;
	//fprintf(drlog, "[Ed]p=%d,l=%d,Ed=%lf\n", p, l, Ed);
	return Ed;
}

void DepthRecoverSolution::disparityCalculation()
{
	int idx0 = indexs[0];
	int rows = pictures[idx0].rgb.rows;
	int cols = pictures[idx0].rgb.cols;
	int pic_num = indexs.size();

	try {
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(cols*rows*pic_num, MAX_K);
		//GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(cols, rows*pic_num,  MAX_K);
		GC = gc;

		//setup up neighborhood system
		for (int p1 = 0; p1 < cols*rows*pic_num; p1++)
		{
			int sn1 = p1 / (rows*cols);
			int idx1 = indexs[sn1];
			int i1_img = (p1 % (rows*cols)) / cols;
			int j1_img = (p1 % (rows*cols)) % cols;

			int p2;
			//left
			if (j1_img > 0)
			{
				p2 = p1 - 1;
				gc->setNeighbors(p1, p2);
			}
			//right
			if (j1_img < cols-1)
			{
				p2 = p1 + 1;
				gc->setNeighbors(p1, p2);
			}
			//up
			if (i1_img > 0)
			{
				p2 = p1 - cols;
				gc->setNeighbors(p1, p2);
			}
			//down
			if (i1_img < rows - 1)
			{
				p2 = p1 + cols;
				gc->setNeighbors(p1, p2);
			}
		}


		// set up the needed data to pass to function for the data costs
		ForDataFn toFn;
		toFn.pictures = &pictures;
		toFn.indexs = &indexs;

		gc->setDataCost(&dataFn_init, &toFn);
		gc->setSmoothCost(&smoothFn, &toFn);

		fprintf(drlog, "--------------------------- compute energy ---------------------------\n");
		double energy_before = gc->compute_energy();
		printf("\nBefore optimization energy is %d\n", energy_before);
		fprintf(drlog, "--------------------------- end of compute energy ---------------------------\n");
		fprintf(drlog,"Before optimization, energy is %lf\n", energy_before);
		gc->expansion(1);
		fprintf(drlog, "--------------------------- end of expansion1 ---------------------------\n");
		gc->setDataCost(&dataFn_iterative, &toFn);
		gc->expansion(2);
		fprintf(drlog, "--------------------------- end of expansion2 ---------------------------\n");
		fprintf(drlog, "--------------------------- compute energy2 ---------------------------\n");
		double energy_after = gc->compute_energy();
		printf("\nAfter optimization energy is %d\n", energy_after);
		fprintf(drlog, "After optimization, energy is %lf\n", energy_after);
		


		for (int i = 0; i < rows*cols*pic_num; i++)
		{
			int l = gc->whatLabel(i);
			double d = (MAX_K - l) / MAX_K * D_min + l / MAX_K * D_max;
			int sn = i / (rows*cols);
			int idx = indexs[sn];
			int i_img = (i % (rows*cols)) / cols;
			int j_img = (i % (rows*cols)) % cols;
			pictures[idx].disparity.at<ushort>(i_img, j_img) = d;
		}
		delete gc;
	}
	catch (GCException e) {
		e.Report();
	}
}

void DepthRecoverSolution::outputdepth(std::string p)
{
	for (int i = 0; i < indexs.size(); i++)
	{
		int idx = indexs[i];
		imwrite((p + pictures[idx].rgb_path).c_str(), pictures[idx].disparity);
	}
}