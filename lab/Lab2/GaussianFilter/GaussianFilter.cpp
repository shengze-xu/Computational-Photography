#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>
#define PI 3.1415926
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	if (argc != 4) {
		cout << "ERROR: Wrong number of parameters." << endl;
		return -1;
	}
	//string in_path = "input.png", out_path = "output.png";
	string in_path, out_path;
	double sigma;

	in_path = argv[1];
	out_path = argv[2];
	sigma = atof(argv[3]);

	int kernel = 2*floor(5 * sigma)+1;
	int center = kernel / 2;
	double sum = 0;

	Mat gauss = Mat(kernel, kernel, CV_64FC1);
	for (int i = 0; i < kernel; i++) {
		for (int j = 0; j < kernel; j++) {
			gauss.at<double>(i, j) = (1 / (2 * PI*sigma*sigma))*exp(-((i - center)*(i - center) + (j - center)*(j - center)) / (2 * sigma*sigma));
			sum += gauss.at<double>(i, j);
		}
	}
	for (int i = 0; i < kernel; i++) {
		for (int j = 0; j < kernel; j++) {
			gauss.at<double>(i, j) /= sum;
		}
	}

	Mat in_img = imread(in_path);
	Mat out_img = imread(in_path);

	filter2D(in_img, out_img, in_img.depth(), gauss);
	imshow("input", in_img); 
	imshow("output", out_img);
	imwrite(out_path, out_img);
	waitKey(0);

	return 0;
}