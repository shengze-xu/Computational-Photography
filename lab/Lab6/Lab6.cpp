#include <iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\xfeatures2d.hpp>
#include<vector>

using namespace std;
using namespace cv;
using namespace xfeatures2d;


class CylindricalPanorama {
public:
	virtual bool makePanorama(vector<Mat>& img_vec, Mat& img_out, double f) = 0;
};

class Panorama2721 : public CylindricalPanorama {
	double r = 500;
public:
	vector<Mat> image;
	void readImage(vector<string> name, double f);
	void OperateImage(vector<Mat>& image, double f);
	bool makePanorama(vector<Mat>& img_vec, Mat& img_out, double f);
	Mat CylinderTransform(Mat image, double r, double f);
	Mat ImageStitch(Mat imageA, Mat imageB);
	void ShowImage();
};

void Panorama2721::ShowImage() {
	for (int i = 0; i < image.size(); i++) {
		imshow(to_string(i), image[i]);
	}
}

void Panorama2721::readImage(vector<string> name, double f) {
	for (int i = 0; i < name.size(); i++) {
		Mat s = imread(name[i]);
		s = CylinderTransform(s, r, f);
		image.push_back(s);
	}
}

void Panorama2721::OperateImage(vector<Mat>& image, double f) {
	int num = image.size();
	for (int i = 0; i < num; i++) {
		image[i] = CylinderTransform(image[i], r, f);
	}
}

bool Panorama2721::makePanorama(vector<Mat>& image, Mat& img_out, double f) {
	int num = image.size();
	//cout << num << endl;
	if (num <= 0) {
		return false;
	}
	int mid = num / 2;
	Mat img = image[mid];
	if (num % 2 == 1) {
		for (int i = 1; i <= mid; i++) {
			img = ImageStitch(img, image[mid - i]);
			img = ImageStitch(img, image[mid + i]);
			imshow("img_now", img);
			waitKey(0);
		}
	}
	else {
		for (int i = 1; i < mid; i++) {
			img = ImageStitch(img, image[mid - i]);
			img = ImageStitch(img, image[mid + i]);
			imshow("img_now", img);
			waitKey(0);
		}
		img = ImageStitch(img, image[0]);
		imshow("img_now", img);
		
	}
	imwrite("img_result1.jpg", img);
	img.copyTo(img_out);
	return true;
}

Mat Panorama2721::CylinderTransform(Mat image,double r,double f){
	int rows = image.rows;
	int cols = image.cols;
	int xmax = (cols - 1) / 2;
	int ymax = (rows - 1) / 2;
	int x1 = r * atan(xmax / f);
	int y1 = r * ymax / f;
	int col1 = x1 * 2 + 1;
	int row1 = y1 * 2 + 1;
	Mat M(row1, col1, CV_8UC3, cv::Scalar::all(1));
	for (int i = 0; i < row1; i++) {
		for (int j = 0; j < col1; j++) {
			double x = f * tan((j - x1) / r);
			double y = (i - y1) / r * sqrt(x*x + f * f);
			x = x + xmax;
			y = y + ymax;
			if (x < 0 || y < 0 || x >= cols || y >= rows) {
				M.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
			else {
				M.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(y, x);
			}
		}
	}
	return M;
}

Mat Panorama2721::ImageStitch(Mat imageA, Mat imageB){
	//imshow("imageA", imageA);
	//imshow("imageB", imageB);
	copyMakeBorder(imageA, imageA, 10, 10, 50, 50, BORDER_CONSTANT, Scalar(0, 0, 0));
	copyMakeBorder(imageB, imageB, 10, 10, 50, 50, BORDER_CONSTANT, Scalar(0, 0, 0));
	//imshow("imageA_1", imageA);
	//imshow("imageB_1", imageB);

	int Hessian = 1000;
	Ptr<SIFT>detector = SIFT::create(Hessian);

	vector<KeyPoint>keypointA, keypointB;
	Mat descriptorA, descriptorB;
	detector->detectAndCompute(imageA, Mat(), keypointA, descriptorA);
	detector->detectAndCompute(imageB, Mat(), keypointB, descriptorB);

	Mat drawsrc, drawsrc2;
	drawKeypoints(imageA, keypointA, drawsrc);
	//imshow("drawsrc", drawsrc);
	//imwrite("drawsrc.jpg", drawsrc);
	drawKeypoints(imageB, keypointB, drawsrc2);
	//imshow("drawsrc2", drawsrc2);
	//imwrite("drawsrc2.jpg", drawsrc2);

	//FlannBasedMatcher matcher;
	BFMatcher matcher(NORM_L2);
	vector<DMatch>matches;
	matcher.match(descriptorA, descriptorB, matches);

	vector<DMatch>goodmatches;
	vector<Point2f>goodkeypointA, goodkeypointB;
	sort(matches.begin(), matches.end());

	for (int i = 0; i < 300; i++){
		goodkeypointA.push_back(keypointA[matches[i].queryIdx].pt);
		goodkeypointB.push_back(keypointB[matches[i].trainIdx].pt);
		goodmatches.push_back(matches[i]);
	}

	//Mat temp_A = imageA.clone();
	//for (int i = 0; i < goodkeypointA.size(); i++){
		//circle(temp_A, goodkeypointA[i], 3, Scalar(0, 255, 0), -1);
	//}
	//imshow("goodkeypoints1", temp_A);

	//Mat temp_B = imageB.clone();
	//for (int i = 0; i < goodkeypointB.size(); i++){
		//circle(temp_B, goodkeypointB[i], 3, Scalar(0, 255, 0), -1);
	//}
	//imshow("goodkeypoints2", temp_B);

	Mat img_matches;
	drawMatches(imageA, keypointA, imageB, keypointB, goodmatches, img_matches);
	//imshow("Match", img_matches);
	//imwrite("Match.jpg", img_matches);

	Mat H = findHomography(goodkeypointB, goodkeypointA, RANSAC);
	
	Mat DstImg;
	warpPerspective(imageB, DstImg, H, Size(imageA.cols + imageB.cols, imageA.rows));
	//copyMakeBorder(imageB, imageB, 0, 0, 0, 200, BORDER_CONSTANT, Scalar(0, 0, 0));
	//imshow("DstImg", DstImg);

	//copyMakeBorder(imageA, imageA, 0, 0, 0, 100, BORDER_CONSTANT, Scalar(0, 0, 0));
	for (int i = 0; i < imageA.rows; i++) {
		for (int j = 0; j < imageA.cols; j++) {
			if (norm(imageA.at<Vec3b>(i, j)) < norm(DstImg.at<Vec3b>(i, j))) imageA.at<Vec3b>(i, j) = DstImg.at<Vec3b>(i, j);
		}
	}
	//imshow("imgA", imageA);
	//imwrite("imgA.jpg", imageA);
	return imageA;
}

vector<string> data1 = {
	"./image/data1/DSC01599.jpg",
	"./image/data1/DSC01600.jpg",
	"./image/data1/DSC01601.jpg",
	"./image/data1/DSC01602.jpg",
	"./image/data1/DSC01603.jpg",
	"./image/data1/DSC01604.jpg",
	"./image/data1/DSC01605.jpg",
	"./image/data1/DSC01606.jpg",
	"./image/data1/DSC01607.jpg",
	"./image/data1/DSC01608.jpg",
	"./image/data1/DSC01609.jpg",
	"./image/data1/DSC01610.jpg",
	"./image/data1/DSC01611.jpg",
	"./image/data1/DSC01612.jpg",
	"./image/data1/DSC01613.jpg",
	"./image/data1/DSC01614.jpg",
	"./image/data1/DSC01615.jpg",
	"./image/data1/DSC01616.jpg",
	"./image/data1/DSC01617.jpg",
	"./image/data1/DSC01618.jpg"
};

vector<string> data2 = {
	"./image/data2/DSC01538.jpg",
	"./image/data2/DSC01539.jpg",
	"./image/data2/DSC01540.jpg",
	"./image/data2/DSC01541.jpg",
	"./image/data2/DSC01542.jpg",
	"./image/data2/DSC01543.jpg",
	"./image/data2/DSC01544.jpg",
	"./image/data2/DSC01545.jpg",
	"./image/data2/DSC01546.jpg",
	"./image/data2/DSC01547.jpg",
	"./image/data2/DSC01548.jpg",
	"./image/data2/DSC01549.jpg",
};

int main() {
	double f = 512.89;
	Panorama2721 myparanoma;
	myparanoma.readImage(data1, f);
	//myparanoma.readImage(data2, f);
	//myparanoma.ShowImage();

	//Mat image = imread("./image/data1/DSC01599.jpg");
	//image = myparanoma.CylinderTransform(image, 500, f);
	//imshow("transform", image);
	//imwrite("transform.jpg", image);

	//Mat image = myparanoma.ImageStitch(myparanoma.image[0], myparanoma.image[1]);
	//imshow("stitch", image);
	//imwrite("stitch.jpg", image);

	Mat img;
	myparanoma.makePanorama(myparanoma.image, img, f);
	//imwrite("img.jpg", img);
	//imshow("img", img);
	waitKey(0);
}


