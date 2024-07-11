#pragma once
#include "ZImageUtil.h"
//#include"ZImageUtil.hpp"
//#include<cv.h>
//#include<highgui.h>
void ZGaussSmooth(const ZFloatImage& srcImg, ZFloatImage& resImg, double sigma=1);