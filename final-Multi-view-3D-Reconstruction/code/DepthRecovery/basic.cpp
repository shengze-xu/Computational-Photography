#include "windows.h"
#include "basic.h"
#include"iostream"
#include<vector>
//#include<cv.h>
//#include<highgui.h>
void ZGaussSmooth(const ZFloatImage& srcImg, ZFloatImage& resImg, double sigma)
{
	int ksize=7;
	if(srcImg.GetChannel()!=1) return;
	sigma=sigma>0?sigma:-sigma;
	//int ksize = 7; //ceil(sigma*3)*2+1;
	if(ksize % 2 == 0)
		++ksize;
	if(ksize==1){
		resImg = srcImg;
		return;
	}
	ZFloatImage temp(srcImg);
	temp.MakeZero();

	std::vector<double> kernel(ksize);
	double scale = -0.5/(sigma*sigma);
	// double PI=3.141592653;
	double cons = 1.0 / sqrt(2 * PI) / sigma;
	double sum=0;
	int kcenter = ksize / 2;

	for(int i=0;i<ksize;i++){
		int x = i-kcenter;
		kernel[i] = cons*exp(x*x*scale);
		sum += kernel[i];
	}
	for(int i=0; i<ksize; i++){
		kernel[i] /= sum;
	}

	//	std::cout<<kernel[i]<<"   "<<std::endl;
	int iHeight = srcImg.GetHeight();
	int iWeight = srcImg.GetWidth();
	for(int y=0; y<iHeight; y++){
		for(int x=0; x<iWeight; x++){
			double mul=0, sum=0;
			//double bmul=0,gmul=0,rmul=0;
			for(int i=-kcenter; i<=kcenter; i++){
				if(x+i >= 0 && x+i < iWeight){
					mul += srcImg.at(x+i,y) * kernel[kcenter+i];
					sum += kernel[kcenter+i];
				}
			}
			//std::cout<<"   "<<mul/sum<<"     "<<std::endl;
			temp.at(x,y) = mul / sum;
		}
	}
	for(int y=0; y<iHeight; y++){
		for(int x=0; x<iWeight; x++){
			double mul=0, sum=0;
			//double bmul=0,gmul=0,rmul=0;
			for(int i = -kcenter; i < kcenter; ++i){
				if(y+i>=0 && y+i<iHeight){
					mul += temp.at(x, y+i) * kernel[kcenter+i];
					sum += kernel[kcenter+i];
				}
			}
			//	 std::cout<<"   "<<mul/sum<<"     "<<std::endl;
			resImg.at(x, y) =  mul / sum;
		}
	}
}

//
// void cvImageShow(IplImage *lhs,char *pname)
// {
// 	cvNamedWindow(pname,1);
// 	cvShowImage(pname,lhs);
// 	cvWaitKey(0);
// }