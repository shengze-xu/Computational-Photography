#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <iostream>
#include "GCoptimization.h"
#include "DepthRecover.h"
#include <stdio.h>
#include<opencv2/opencv.hpp>
FILE *drlog;
using namespace std;
using namespace cv;

int main()
{
	fopen_s(&drlog, "D:/log/drlog.txt", "w");
	string path = "D:/cp/testset/teddy/";
	DepthRecoverSolution drs;
	drs.loadInfo(path);
	drs.loadRGBImgAndCreateDisparity();
	drs.disparityCalculation();
	fclose(drlog);

	drs.outputdepth("D:/cp/testset/teddy/depthfromrgb/");

	return 0;
}