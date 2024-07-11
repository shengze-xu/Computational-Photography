#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <math.h>
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
	if (argc != 5) {
		cout << "ERROR: Wrong number of parameters." << endl;
		return -1;
	}
	string in_path, out_path;
	int w, h;

	in_path = argv[1];
	out_path = argv[2];
	w = atoi(argv[3]);
	h = atoi(argv[4]);

	Mat in_img = imread(in_path);
	Mat out_img = imread(in_path);

	int row = in_img.rows;
	int col = in_img.cols;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			vector<vector<int>> color(3);
			Vec3b pixel;
			for (int x = i - h + 1; x <= i + h - 1; x++) {
				for (int y = j - w + 1; y <= j + w - 1; y++) {
					pixel = in_img.at<Vec3b>((x + row) % row, (y + col) % col);
					for (int k = 0; k < 3; k++) {
						color[k].push_back(pixel[k]);
					}
				}
			}
			for (int k = 0; k < 3; k++) {
				sort(color[k].begin(), color[k].end());
			}
			int median = color[0].size() / 2;
			out_img.at<Vec3b>(i, j) = Vec3b(color[0][median], color[1][median], color[2][median]);
		}
	}

	imshow("in", in_img);
	imshow("out", out_img);
	imwrite(out_path, out_img);
	waitKey(0);

	return 0;
}