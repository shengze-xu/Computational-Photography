#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>
using namespace std;
using namespace cv;
#define Pi 3.1415926

// get gaussian value
double Gauss(double sigma, double x){
	return exp(-pow(x, 2) / (2 * pow(sigma, 2))) / (sigma*pow(2 * Pi, 0.5));
}

Vec3b color(Mat in_img, int y, int x) {
	return in_img.at<Vec3b>((y + in_img.rows) % in_img.rows, (x + in_img.cols) % in_img.cols);
}

int main(int argc, char* argv[]) {
	if (argc != 6) {
		cout << "ERROR: Wrong number of parameters." << endl;
		return -1;
	}
	//string in_path = "input.png", out_path = "output.png";
	string in_path, out_path;
	double sigmas, sigmar;
	int kernel;

	in_path = argv[1];
	out_path = argv[2];
	sigmas = atof(argv[3]);
	sigmar = atof(argv[4]);
	kernel = atoi(argv[5]);

	Mat in_img = imread(in_path);
	Mat out_img = imread(in_path);

	int row = in_img.rows;
	int col = in_img.cols;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			for (int k = 0; k < 3; k++) {
				double Wp = 0;
				double Sum = 0;
				for (int y = i - kernel / 2; y < i + kernel / 2 + 1; y++) {
					for (int x = j - kernel / 2; x < j + kernel / 2 + 1; x++) {
						double d = sqrt(pow(x - j, 2) + pow(y - i, 2));
						double Id = abs(color(in_img, i, j)[k] - color(in_img, y, x)[k]);
						Wp += Gauss(sigmas, d)*Gauss(sigmar, Id);
						Sum += Gauss(sigmas, d)*Gauss(sigmar, Id)*color(in_img, y, x)[k];
					}
				}
				out_img.at<Vec3b>(i, j)[k] = Sum / Wp;
			}
		}
	}

	imshow("in", in_img);
	imshow("out", out_img);
	imwrite(out_path, out_img);

	waitKey(0);

	return 0;
}