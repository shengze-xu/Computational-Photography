#include "PixelCostComputorBase.h"

CPixelCostComputorBase::CPixelCostComputorBase(){
	m_pNearFrames = nullptr;
	m_pCurrentFrame = nullptr;
	m_fDataCostWeight = 20;
	m_dColorMissPenalty = 30;
}

void CPixelCostComputorBase::SetFrames( CMDRVideoFrame *CurrentFrame, std::vector<CMDRVideoFrame*> *nearFrames ){
	m_pCurrentFrame = CurrentFrame;
	m_pNearFrames = nearFrames;
}

void CPixelCostComputorBase::SetSigma( double dColorSigma, double dDspSigma ){
	m_dColorSigma = dColorSigma;
	m_dDspSigma = dDspSigma;
}

CPixelCostComputorBase::~CPixelCostComputorBase( void )
{
}