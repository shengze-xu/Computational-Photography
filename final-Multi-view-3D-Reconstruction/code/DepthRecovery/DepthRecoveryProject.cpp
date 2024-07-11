// DepthRecoveryProject.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <afx.h>
#include "parser/VDRStructureMotion.h"
#include "Parser/TrackFileParser.h"
#include <iostream>

int _tmain(int argc, _TCHAR* argv[])
{
	CVDRStructureMotion sfm;
	CTrackFileParser parser(&sfm);
	if(!parser.LoadProject( "/*actsÎÄ¼þÄ¿Â¼*/ ")){
		system("pause");
		return 1;
	}
	for(int k=0; k<sfm.GetFrameCount(); ++k){
		CVDRVideoFrame* pFrame = sfm.GetFrameAt(k);
		pFrame->m_pvMatchPoints = parser.m_vpFramesMatchPoints[k];
	}

	double dspMax = sfm.EstimateDisparityRange(parser);

	int end=sfm.GetFrameCount()-1;
	sfm.RecoverDepth(0,end+1);

	system("pause");
	return 0;
}