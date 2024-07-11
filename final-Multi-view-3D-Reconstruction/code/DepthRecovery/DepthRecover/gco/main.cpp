//////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////
//
//  Optimization problem:
//  is a set of sites (pixels) of width 10 and hight 5. Thus number of pixels is 50
//  grid neighborhood: each pixel has its left, right, up, and bottom pixels as neighbors
//  7 labels
//  Data costs: D(pixel,label) = 0 if pixel < 25 and label = 0
//            : D(pixel,label) = 10 if pixel < 25 and label is not  0
//            : D(pixel,label) = 0 if pixel >= 25 and label = 5
//            : D(pixel,label) = 10 if pixel >= 25 and label is not  5
// Smoothness costs: V(p1,p2,l1,l2) = min( (l1-l2)*(l1-l2) , 4 )
// Below in the main program, we illustrate different ways of setting data and smoothness costs
// that our interface allow and solve this optimizaiton problem

// For most of the examples, we use no spatially varying pixel dependent terms. 
// For some examples, to demonstrate spatially varying terms we use
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with 
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "GCoptimization.h"


int Num_Pixels;
int Num_Labels;
GCoptimizationGeneralGraph *GC;

struct MYDCF: public GCoptimization::DataCostFunctor {
	GCoptimization::EnergyTermType compute(GCoptimization::SiteID s, GCoptimization::LabelID l);
}mydcf;

GCoptimization::EnergyTermType MYDCF::compute(GCoptimization::SiteID s, GCoptimization::LabelID l)
{
    printf("%d %d\n",GC->whatLabel(s),l);
    if (s < 25 ){
        if(  l == 0 ) return 0;
        else return 10;
    }
    else {
        if(  l == 5 ) return 0;
        else return 10;
    }
}

int smoothFn(int p1, int p2, int l1, int l2)
{
	if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	else return(4);
}

void GeneralGraph_DataCostFunctor(int width,int height,int num_pixels,int num_labels)
{

    int *result = new int[num_pixels];
    try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(width*height,num_labels);
        GC = gc;
		
        for(int i=0;i<height;i++)
        {
            for(int j=0;j<width;j++)
            {
                int p = i*width+j;
                if(j!=width-1)
                {
                    int right=p+1;
                    gc->setNeighbors(p,right);
                }
                if(i!=height-1)
                {
                    int below = p+width;
                    gc->setNeighbors(p,below);
                }
            }
        }

		gc->setDataCostFunctor(&mydcf);
		gc->setSmoothCost(&smoothFn);

		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

    delete [] result;
}
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	int width = 10;
	int height = 5;
	int num_pixels = width*height;
	int num_labels = 7;
    Num_Pixels = num_pixels;
    Num_Labels = num_labels;

    GeneralGraph_DataCostFunctor(width,height,num_pixels,num_labels);

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////

