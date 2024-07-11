#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	if (argc != 5) {
		cout << "ERROR: Wrong number of parameters." << endl;
		return -1;
	}

	string in_path, out_path;
	int w, h, kw, kh;

	in_path = argv[1];
	out_path = argv[2];
	w = atoi(argv[3]);
	h = atoi(argv[4]);

	//The size of the convolution kernel
	kw = 2 * w + 1;
	kh = 2 * h + 1;

	Mat in_img = imread(in_path);
	Mat out_img = imread(in_path);
	Mat kernel = (Mat_<double>(kw, kh));
	for (int i = 0; i < kw; i++) {
		for (int j = 0; j < kh; j++) {
			kernel.at<double>(i, j) = 1.0 / (kw * kh);
		}
	}

	filter2D(in_img, out_img, in_img.depth(), kernel);
	imshow("input", in_img);
	imshow("output", out_img);
	imwrite(out_path, out_img);
	waitKey(0);

	return 0;
}