#pragma once
#include <vector>
#include<iostream>
#include"ZImageUtil.h"
class CDataCostUnit;
class CMDRVideoFrame;

class CPixelCostComputorBase
{
public:
	CPixelCostComputorBase();
	virtual ~CPixelCostComputorBase(void);

	virtual void SetFrames(CMDRVideoFrame *CurrentFrame, std::vector<CMDRVideoFrame*> *nearFrames);
	void SetSigma(double dColorSigma, double dDspSigma);

	virtual void PixelDataCost(std::vector<std::shared_ptr<ZFloatImage>>& m_MiCost,std::vector<float*> &m_prow,std::vector<float*>& m_pcol,int x, int y, const std::vector<double> &dspV, CDataCostUnit &dataCosti, int &bestLabel) = 0;

	double GetDspSigma(){return m_dDspSigma;}
	void SetDspSigma(double sigma){m_dDspSigma = sigma;}
	double GetColorSigma(){return m_dColorSigma;}
	void SetColorSigma(double simga){m_dColorSigma = simga;}
	float GetDataCostWeight(){return m_fDataCostWeight;}
	void SetDataCostWeight(float weight){m_fDataCostWeight = weight;}
	int  GetFramesCount(){return m_pNearFrames->size();}
protected:
	std::vector<CMDRVideoFrame*> *m_pNearFrames;
	CMDRVideoFrame *m_pCurrentFrame;

	double m_dDspSigma;
	double m_dColorSigma;
	double m_dColorMissPenalty;

	float m_fDataCostWeight;
};