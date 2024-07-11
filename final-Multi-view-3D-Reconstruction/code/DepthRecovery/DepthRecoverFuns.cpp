#include "stdafx.h"

#include "DepthRecoverFuns.h"
#include "cximage.h"
#include "BeliefPropagation.h"
#include "LevenMarOptimizer.h"

#include "InitByColorWorkUnit.h"
#include "BORefineWorkUnit.h"
#include "SubRefineWorkUnit.h"
#include "PlanFittingWorkUnit.h"
#include "ParallelManager.h"
#include "WmlMatrix4.h"

#include "GPU/GpuMemManage.h"
#include "GPU/GpuGetDataCost.h"
#include "gpu/GpuBP.h"

//#include "./MRF/mrf.h"
//#include "./MRF/GCoptimization.h"
#include "./GCO/GCoptimization.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <queue>
#include"MiDataCostWorkUnit.h"

#include "TrainingDataGeneratorAndValidator.h"
#include "RandomForestSingleAttribute.h"
#include "ArffParser.h"

const int NeighborCount = 4;
const int Neighbor[NeighborCount][2] = { {0,-1}, {1,0}, {0,1}, {-1,0} };
ZIntImage segm[11];

//struct SPoint{
//	int x;
//	int y;
//	bool operator < (const SPoint &a) const
//	{
//		if(y != a.y)
//			return y<a.y;
//		return x < a.x;
//	}
//};

bool SizeGreater(std::vector< Wml::Vector2<int> >* a, std::vector< Wml::Vector2<int> >* b)
{
	return a->size() > b->size();
}

CDepthRecoverFuns CDepthRecoverFuns::m_Instance;

CDepthRecoverFuns::CDepthRecoverFuns(void)
{
	m_sProjectFilePath = "";

	SetBlockPara(2, 1, 8);

	SetReduceScale(1.0);

	m_ifGenerateTempResult = true;

	m_iStartDst = 1;
	m_iInitStep = 2;
	m_iInitFrames = 4;
	m_iNormalStep = 2;
	m_iMaxFrames = 8;

	m_iDspLevel = 101;
	m_dDspMin = 1.0e-7;
	m_dDspMax = 0.01;
	m_fDspV.resize(m_iDspLevel);
	int layercount = m_iDspLevel - 1;
	for(int i=0;i<m_iDspLevel;i++)
		m_fDspV[i] = m_dDspMin * (layercount-i)/layercount + m_dDspMax * (i)/layercount;

	m_bAutoEstimateDsp = true;

	m_fDataCostWeight = 10;

	m_iRefinePass = 3;	//默认两遍
	m_dColorSigma = 5;
	m_dDspSigma = 0.03;
	m_fColorMissPenalty = 30;

	m_iSubIter = 2;  //默认两遍
	m_iSubSample = 10;
	m_dMinDspSigma = 0.001;

	m_fDiscK = 10;

	m_fSegErrRateThreshold = 5;
	m_fSegSpatial = 5;
	m_fSegColor = 5;
	m_fSegMinsize = 50;
	m_fPlaneFittingSize = 300;

	m_iCpuThreads = 8;
	m_bUseGpu = false;
	m_bUseMI=false;
	m_RunType = RUNTYPE::RUNALL;

	m_iBopfStart=2;
	m_iBopfEnd=2;

	m_iSplitOverlap=0;
 	m_bUseSC=false;
 	m_bUsePriorPlane=false;
	//m_iStartDst = 1;
	//m_iInitStep = 2;
	//m_iInitFrames = 4;
	//m_iNormalStep = 2;
	//m_iMaxFrames = 8;
	//////m_iMaxFrames = 10;

	////AngkorWat
	////m_dDspMin = 1.0e-7;
	////m_dDspMax = 0.00567717;
	////m_iDspLevel = 101;

	////Wall
	////m_dDspMin = 1.0e-7;
	////m_dDspMax = 0.0112894;
	////m_iDspLevel = 101;

	////road
	//m_dDspMin = 1.0e-7;
	//m_dDspMax = 0.015997;
	//m_iDspLevel = 101;

	////m_fSegmaC = 10.0F; ==============================
	////m_fMissPenalty = 50.0F;===========================
	//m_fSegmaC = 5.0F; ==================================
	//m_fMissPenalty = 10.0F;==========================

	//m_fDataCostWeight = 20.0 / 100.0 * (m_iDspLevel - 1);

	////BP
	//m_fDiscK = 10 / 100.0 * (m_iDspLevel - 1); // 10
	//m_fSegErrRateThreshold = 2.5 * m_fDataCostWeight / m_fDiscK;
	//m_fSegSpatial = 5.0F;
	//m_fSegColor = 5.0F;
	//m_fSegMinsize = 50.0F;
	//m_fPlaneFittingSize = 300;

	//m_dColorSigma = 5.0F;
	//m_dPrjSigma = 2.0F;  //3.0
	//m_dDspSigma = 0.03
	//m_fColorMissPenalty = 30.0F;==================
	//m_iSearchWin = 1;

	////==================================================

	//m_iSubSample = 10;
	////float m_fMinDspSigma;
	//m_dMinPrjSigma = 0.5;
	//m_dMinDspSigma = 0.001
	////==================================================

	////threads and gpu
	//m_iCpuThreads = 8;

	//m_dspV.resize(m_iDspLevel);
	//int layercount = m_iDspLevel - 1;
	//for(int i=0;i<m_iDspLevel;i++)
	//	m_dspV[i] = m_dDspMin * (layercount-i)/layercount + m_dDspMax * (i)/layercount;

	//////////////////////////////////////////////////////////////////////////

	//m_iBlockCountX = 2;
	//m_iBlockCountY = 2;
}

CDepthRecoverFuns::~CDepthRecoverFuns(void)
{
}

void CDepthRecoverFuns::SetProjectFilePath(istringstream &in){
	in >> m_sProjectFilePath;
	TCHAR FullPath[MAX_PATH];
	TCHAR PrjDir[MAX_PATH];
	GetFullPathName(m_sProjectFilePath.c_str(), MAX_PATH, FullPath, (TCHAR**)&PrjDir);
	m_sProjectFilePath = FullPath;
}
void CDepthRecoverFuns::SetProjectFilePath( const string& sProjectFilePath )
{
	m_sProjectFilePath = sProjectFilePath;
	TCHAR FullPath[MAX_PATH];
	TCHAR PrjDir[MAX_PATH];
	GetFullPathName(m_sProjectFilePath.c_str(), MAX_PATH, FullPath, (TCHAR**)&PrjDir);
	m_sProjectFilePath = FullPath;
}

void CDepthRecoverFuns::GetNearFrameIndex( std::vector<int>& FwFrameIndex, std::vector<int>& BwFrameIndex,int CurrentIndex, int StartIndex, int EndIndex )
{
	FwFrameIndex.clear();
	BwFrameIndex.clear();

	int iCount = 0;
	for(int index = CurrentIndex + m_iStartDst; index <= EndIndex && iCount < m_iInitFrames; index += m_iInitStep){
		FwFrameIndex.push_back(index);
		++iCount;
	}

	if(iCount > 0){
		for(int index = FwFrameIndex[iCount-1] + m_iNormalStep; index <= EndIndex && iCount < m_iMaxFrames; index += m_iNormalStep){
			FwFrameIndex.push_back(index);
			++iCount;
		}
	}

	iCount = 0;
	for(int index = CurrentIndex - m_iStartDst; index >= StartIndex && iCount < m_iInitFrames; index -= m_iInitStep){
		BwFrameIndex.push_back(index);
		++iCount;
	}

	if(iCount > 0){
		for(int index = BwFrameIndex[iCount-1] - m_iNormalStep; index >= StartIndex && iCount < m_iMaxFrames; index -= m_iNormalStep){
			BwFrameIndex.push_back(index);
			++iCount;
		}
	}

	printf("FW Frames:\t");
	for(int i=0; i<FwFrameIndex.size(); ++i)
		printf("%d\t",FwFrameIndex[i]);
	printf("\n");

	printf("BW Frames:\t");
	for(int i=0; i<BwFrameIndex.size(); ++i)
		printf("%d\t",BwFrameIndex[i]);
	printf("\n");
}

void  CDepthRecoverFuns::GetPList(std::vector<Wml::Matrix4d>& PList, CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& nearFrames)
{
	//Wml::Matrix4d refP, TransposeP;
	//Wml::Matrix3d refK = pCurrentFrame->m_cCameraParameter->m_K;
	//pCurrentFrame->m_cCameraParameter->GetP(refP);

	//if(PList.size() != nearFrames.size())
	//	PList.resize(nearFrames.size());

	//for (int s = 0; s < nearFrames.size(); s++)
	//{
	//	CVideoFrame* pTmpFrame = nearFrames[s];
	//	pTmpFrame->m_cCameraParameter->GetTransposeP(TransposeP);
	//	Wml::Matrix4d P = TransposeP * refP;
	//	Wml::Matrix4d KP = P;
	//	Wml::Matrix3d K = pTmpFrame->m_cCameraParameter->m_K;

	//	for(int i=0;i<3;i++){
	//		for(int j=0;j<4;j++){
	//			KP(i,j) = 0.0;
	//			for(int k=0;k<3;k++){
	//				KP(i,j) += K(i,k)*P(k,j);
	//			}
	//		}
	//	}
	//	PList[s] = KP;
	//}

	Wml::Matrix4d refP;
	pCurrentFrame->GetCameraParaP(refP);

	Wml::Matrix4d TransposeP;
	Wml::Matrix4d P ;
	Wml::Matrix4d KP ;
	Wml::Matrix3d K ;

	if(PList.size() != nearFrames.size())
		PList.resize(nearFrames.size());

	for (int s = 0; s < nearFrames.size(); s++)
	{
		CVDRVideoFrame* pTmpFrame = nearFrames[s];
		pTmpFrame->GetTransposeP(TransposeP);
		P = TransposeP * refP;
		KP = P;
		pTmpFrame->GetCameraInternalPara(K);

		for(int i=0;i<3;i++){
			for(int j=0;j<4;j++){
				KP(i,j) = 0.0;
				for(int k=0;k<3;k++){
					KP(i,j) += K(i,k)*P(k,j);
				}
			}
		}
		PList[s] = KP;
	}
}

//void  CDepthRecoverFuns::GetPListAndPInvList(std::vector<Wml::Matrix4d>& PList, std::vector<Wml::Matrix4d>& PInvList, CVideoFrame* pCurrentFrame, std::vector<CVideoFrame*>& nearFrames)
//{
	//Wml::Matrix4d refP, TransposeP;
	//Wml::Matrix3d refK = pCurrentFrame->m_cCameraParameter->m_K;
	//pCurrentFrame->m_cCameraParameter->GetP(refP);

	//if(PList.size() != nearFrames.size())
	//	PList.resize(nearFrames.size());
	//if(PInvList.size() != nearFrames.size())
	//	PInvList.resize(nearFrames.size());

	//for (int s = 0; s < nearFrames.size(); s++)
	//{
	//	CVideoFrame* pTmpFrame = nearFrames[s];
	//	pTmpFrame->m_cCameraParameter->GetTransposeP(TransposeP);
	//	Wml::Matrix4d P = TransposeP * refP;

	//	Wml::Matrix4d KP = P;
	//	Wml::Matrix3d K = pTmpFrame->m_cCameraParameter->m_K;

	//	Wml::Matrix4d PInv = P;
	//	CCameraParameters::InverseP(PInv);

	//	Wml::Matrix4d KPInv = PInv;

	//	for(int i=0;i<3;i++){
	//		for(int j=0;j<4;j++){
	//			KP(i,j) = 0.0;
	//			KPInv(i,j) = 0.0;
	//			for(int k=0;k<3;k++){
	//				KP(i,j) += K(i,k)*P(k,j);
	//				KPInv(i,j) += refK(i,k)*PInv(k,j);
	//			}
	//		}
	//	}
	//	PList[s] = KP;
	//	PInvList[s] = KPInv;
	//}
//}

void CDepthRecoverFuns::GetInitialDataCost( std::vector<CVDRVideoFrame*>& nearFrames, std::vector<Wml::Matrix4d>& PList,CVDRVideoFrame* pCurrentFrame, CDataCost& outDataCost, ZIntImage& labelImg )
{
	CParallelManager pm(m_iCpuThreads);
	for(int j=0; j<m_iBlockHeight; ++j){
		CInitByColorWorkUnit* pWorkUnit = new CInitByColorWorkUnit(pCurrentFrame, nearFrames, PList, labelImg, outDataCost, j, m_iBlockWidth);//,dspV,depthEst,labelImg,j);
		pm.EnQueue(pWorkUnit);
	}
	pm.Run();
	//GetInitialDataCostAt(pCurrentFrame,u, v, dataCosti, nearFrames, labelImg, PList);
}

void CDepthRecoverFuns::GetInitialDataCostAt( CVDRVideoFrame* pCurrentFrame, int u, int v, CDataCostUnit& dataCosti, std::vector<CVDRVideoFrame*>& NearFrames,
											 int& bestLabel, std::vector<float>& dspV, std::vector<Wml::Matrix4d>& PList)
{
	double MaxLikelihood = 1e-6F;
	int frameCount = NearFrames.size();
	int dspLevelCount = dspV.size();

	Wml::Vector3d ptWorldCoord;
	Wml::Vector3d CurrentColor, CorrespondingColor;
	double z, u2, v2, z2, dsp;

	vector< vector<double> > r (NearFrames.size(), vector<double>(3));
	vector< vector<double> > b (NearFrames.size(), vector<double>(3));
	double fx = (u - pCurrentFrame->GetInternalParaAt(0,2))/pCurrentFrame->GetInternalParaAt(0,0);
	double fy = (v - pCurrentFrame->GetInternalParaAt(1,2))/pCurrentFrame->GetInternalParaAt(1,1);
	for(int i=0; i<NearFrames.size(); i++){
		NearFrames[i]->Calculate(&r[i][0], &b[i][0], fx, fy, PList[i]);
	}
	int realFrameCount=0;
	pCurrentFrame->GetColorAt(u, v, CurrentColor);
	for(int depthLeveli = 0; depthLeveli < dspLevelCount; depthLeveli++){
		dataCosti[depthLeveli] = 0;
		realFrameCount=0;
		//pCurrentFrame->m_cCameraParameter->GetWorldCoordFrmImgCoord(u, v, dspV[depthLeveli], ptWorldCoord, true);
		z = 1.0/dspV[depthLeveli];
		for(int i=0; i<frameCount; i++)
		{
			//NearFrames[i]->m_cCameraParameter->GetImgCoordFrmWorldCoord(u2, v2, dsp, ptWorldCoord, true);
			z2 = r[i][2]*z + b[i][2];
			u2 = (r[i][0]*z + b[i][0]) / z2;
			v2 = (r[i][1]*z + b[i][1]) / z2;

			if(u2<0 || v2<0 || u2>= CVDRVideoFrame::GetImgWidth() || v2>= CVDRVideoFrame::GetImgHeight()) ;
			//	dataCosti[depthLeveli] += m_dColorSigma / (m_dColorSigma + (m_fColorMissPenalty/3)*(m_fColorMissPenalty/3));
			else
			{
				 NearFrames[i]->GetColorAt(u2, v2, CorrespondingColor);

				double colordist = (fabs(CurrentColor[0] - CorrespondingColor[0])
					+ fabs(CurrentColor[1] - CorrespondingColor[1])
					+ fabs(CurrentColor[2] - CorrespondingColor[2])) / 3.0;

				colordist *= colordist;
				dataCosti[depthLeveli] += m_dColorSigma / (m_dColorSigma + colordist);
				realFrameCount++;
			}
		}
		if(realFrameCount!=0)
			dataCosti[depthLeveli]/=realFrameCount;
		else dataCosti[depthLeveli]=0;
		if(dataCosti[depthLeveli] > MaxLikelihood){
			MaxLikelihood = dataCosti[depthLeveli];
			bestLabel = depthLeveli;
			//labelImg.SetPixel(u, v, 0, depthLeveli);
		}
	}

	//Normalize
	float maxCost = 1e-6F;
	for(int depthLeveli = 0; depthLeveli < dspLevelCount; depthLeveli++){
		dataCosti[depthLeveli] =  1.0 - dataCosti[depthLeveli] / MaxLikelihood;
		maxCost = max(maxCost,dataCosti[depthLeveli]);
	}

	float dataCostWeight = GetTrueDataCostWeight();
	for(int depthLeveli=0; depthLeveli<dspLevelCount; depthLeveli++){
		dataCosti[depthLeveli] /= maxCost;
		dataCosti[depthLeveli] *= dataCostWeight;
//		std::cout<<"$"<<dataCosti[depthLeveli]<<std::endl;
	}
}
void CDepthRecoverFuns::GetInitialDataCostAt( CVDRVideoFrame* pCurrentFrame,int u, int v, CDataCostUnit& dataCosti,
											 std::vector<CVDRVideoFrame*>& NearFrames, int& bestLabel, std::vector<Wml::Matrix4d>& PList){
	GetInitialDataCostAt(pCurrentFrame, u, v, dataCosti, NearFrames, bestLabel, m_fDspV, PList);
}

void CDepthRecoverFuns::GetBundleDataCost( std::vector<CVDRVideoFrame*>& nearFrames, std::vector<Wml::Matrix4d>& PList,
										   CVDRVideoFrame* pCurrentFrame, CDataCost& outDataCost, ZIntImage& labelImg)
{
	CParallelManager pm(m_iCpuThreads);
	for(int j=0; j<m_iBlockHeight; ++j){
		CBORefineWorkUnit* pWorkUnit = new CBORefineWorkUnit(pCurrentFrame, nearFrames, PList,m_fDspV, labelImg, outDataCost, j, m_iBlockWidth);
		pm.EnQueue(pWorkUnit);
	}
	pm.Run();
}

void CDepthRecoverFuns::GetBundleDataCostAt( CVDRVideoFrame* pCurrentFrame,int u, int v, CDataCostUnit& dataCosti,
											 std::vector<CVDRVideoFrame*>& NearFrames, int& bestLabel, std::vector<float>& dspV,
											 std::vector<Wml::Matrix4d>& PList)
{
	double MaxLikelihood = 1.0e-6F;
	int frameCount = NearFrames.size();
	int dspLevelCount = dspV.size();

	//Wml::Vector3d ptWorldCoord, ptWorldCoord2;
	Wml::Vector3d CurrentColor, CorrespondingColor;
	double z, u2, v2, dsp, u21, v21, z2;

	vector< vector<double> > r (NearFrames.size(), vector<double>(3));
	vector< vector<double> > b (NearFrames.size(), vector<double>(3));
	double fx = (u - pCurrentFrame->GetInternalParaAt(0,2))/pCurrentFrame->GetInternalParaAt(0,0);
	double fy = (v - pCurrentFrame->GetInternalParaAt(1,2))/pCurrentFrame->GetInternalParaAt(1,1);
	for(int i=0; i<NearFrames.size(); i++){
		NearFrames[i]->Calculate(&r[i][0], &b[i][0], fx, fy, PList[i]);
	}

	pCurrentFrame->GetColorAt(u, v, CurrentColor);

	for(int depthLeveli = 0; depthLeveli < dspLevelCount; depthLeveli++){
		dataCosti[depthLeveli]  = 0;
		//pCurrentFrame->m_cCameraParameter->GetWorldCoordFrmImgCoord(u, v, dspV[depthLeveli], ptWorldCoord, true);
		z = 1.0/dspV[depthLeveli];

		for(int i=0; i<frameCount; i++)
		{
			//NearFrames[i]->m_cCameraParameter->GetImgCoordFrmWorldCoord(u2, v2, dsp, ptWorldCoord, true);
			//z2 = 1.0/dsp;

			z2 = r[i][2]*z + b[i][2];
			u2 = (r[i][0]*z + b[i][0]) / z2;
			v2 = (r[i][1]*z + b[i][1]) / z2;

			if(u2<0.1||u2>= CVDRVideoFrame::GetImgWidth()-1.1 || v2<0.1||v2>= CVDRVideoFrame::GetImgHeight()-1.1){
				//dataCost[depthLeveli] += m_dColorSigma / (m_dColorSigma + m_fColorMissPenalty) * exp(-6.0*6.0/(2.0*3.0*3.0));
				dataCosti[depthLeveli] += m_dColorSigma / (m_dColorSigma + m_fColorMissPenalty) * 0.01;
			}
			else
			{
				NearFrames[i]->GetColorAt(u2, v2, CorrespondingColor);
				float colordist = (fabs(CurrentColor[0] - CorrespondingColor[0])
					+ fabs(CurrentColor[1] - CorrespondingColor[1])
					+ fabs(CurrentColor[2] - CorrespondingColor[2])) / 3.0F;

				colordist = min(30.0, (double)colordist);

				//double wc = m_dColorSigma/(m_dColorSigma + colordist);
				double wc = (m_dColorSigma*m_dColorSigma)/(m_dColorSigma*m_dColorSigma + colordist*colordist);

				double d2 = NearFrames[i]->GetDspAt(u2, v2);
				double d2_INT = NearFrames[i]->GetDspAt((int)(u2+0.5F), (int)(v2+0.5F));
				dsp = 1.0/z2;

				if(  fabs(d2_INT - dsp) < fabs(d2 - dsp) ){
					d2 = d2_INT;
				}

				//NearFrames[i]->m_cCameraParameter->GetWorldCoordFrmImgCoord(u2, v2, d2, ptWorldCoord2);
				//pCurrentFrame->m_cCameraParameter->GetImgCoordFrmWorldCoord(u21, v21, dsp, ptWorldCoord2);
				//double dist2 = (u21-u)*(u21-u) + (v21-v)*(v21-v);
				////double wd = exp(-dist2 / (2.0* m_dPrjSigma * m_dPrjSigma));
				//double wd = max(1e-3, exp(-dist2/(2.0*m_dPrjSigma*m_dPrjSigma)));

				//////////////////////////////////////////////////////////////////////////
				//double z12 = 1.0/d2;
				//Wml::Matrix3d& K = NearFrames[i]->m_cCameraParameter->m_K;
				//double x12 = (u2 - K(0,2))/K(0,0) * z12;
				//double y12 = (v2 - K(1,2))/K(1,1) * z12;

				//Wml::Matrix4d& PInv = PInvList[i];
				//double x21 = x12*PInv(0,0) + y12*PInv(0,1) + z12*PInv(0,2) + PInv(0,3);
				//double y21 = x12*PInv(1,0) + y12*PInv(1,1) + z12*PInv(1,2) + PInv(1,3);
				//double z21 = x12*PInv(2,0) + y12*PInv(2,1) + z12*PInv(2,2) + PInv(2,3);

				//if(z21 == 0.0){
				//	printf("z21 == 0, so divide zero!");
				//}
				//x21 /= z21;
				//y21 /= z21;

				//double dist2 = (x21-u)*(x21-u) + (y21-v)*(y21-v);
				//double wd = max(1e-3,exp(-dist2/(2.0*m_dPrjSigma*m_dPrjSigma)));
				//////////////////////////////////////////////////////////////////////////

				//====================================================================
				double dspDiff = fabs(d2 - dsp);
				double dspSigma = (m_dDspMax - m_dDspMin) * m_dDspSigma;
				double dspSigma2 = dspSigma * dspSigma;
				double wd = dspSigma2 / (dspSigma2 + dspDiff*dspDiff*2);
				//====================================================================
			//	dataCosti[depthLeveli] += max(1e-6, wd);
				///////////////////////////////////////////////////////////////////////////////////////////////////////
		//		std::cout<<"wc="<<wc<<"  ,wd="<<wd<<std::endl;

				dataCosti[depthLeveli] += max(1e-6, wc * wd);
			}
		}
		dataCosti[depthLeveli] /= frameCount;
		if(dataCosti[depthLeveli] > MaxLikelihood){
			MaxLikelihood = dataCosti[depthLeveli];
			bestLabel = depthLeveli;
			//labelImg.SetPixel(u, v, 0, depthLeveli);
		}
	}

	//Normalize
	float maxCost = 1.0e-6F;
	for(int depthLeveli = 0; depthLeveli < dspLevelCount; depthLeveli++){
		dataCosti[depthLeveli] =  1.0 - dataCosti[depthLeveli] / MaxLikelihood;
		maxCost = max(maxCost,dataCosti[depthLeveli]);
	}

	float dataCostWeight = GetTrueDataCostWeight();
	for(int depthLeveli=0; depthLeveli<dspLevelCount; depthLeveli++){
		dataCosti[depthLeveli] /= maxCost;
		dataCosti[depthLeveli] *= dataCostWeight;
	}
}

void CDepthRecoverFuns::RefineDspByBP( CDataCost& DataCost, ZIntImage& labelImg, bool addEdgeInfo, ZIntImage* offsetImg)
{
	CBeliefPropagation bp(
		m_ImgPartition,
		GetTrueDiscK(),		//disc_k
//	       100.0,
		DataCost.GetChannel() ,   //max_d
		0.1F,			//sigma for gauss smooth
		offsetImg != NULL ? 10 : 5,				//nIter
		offsetImg != NULL ? 1 : 5,				//nlevels
		m_iCpuThreads	//nThreads
		);

	bp.printPara();
	bp.stereoMatching(m_iBlockIdx, m_iBlockIdy, DataCost, labelImg, addEdgeInfo, offsetImg);
}

void CDepthRecoverFuns::RefineDspByGpuBp(CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr, ZIntImage& labelImg, void* d_labelImg, void* offsetImg, size_t d_LabelImgPitch, bool addEdgeInfo)
{
	SBPParas paras;
	paras.blockWidth = m_iBlockWidth;
	paras.blockHeight = m_iBlockHeight;
	paras.disc_k = GetTrueDiscK();
	paras.max_d = DataCost.GetChannel() ;
	paras.nBpLevels = offsetImg != NULL ? 1 : 5;
	paras.nIter = offsetImg != NULL ? 10 : 5;
	paras.offsetX = m_iOffsetX;
	paras.offsetY = m_iOffsetY;
	paras.sigma = 0.1F;
	paras.trueX1 = m_iTrueX1;
	paras.trueY1 = m_iTrueY1;
	paras.trueX2 = m_iTrueX2;
	paras.trueY2 = m_iTrueY2;

	int ImgWidth = labelImg.GetWidth();
	int ImgHeight = labelImg.GetHeight();

	printf("BP: MaxD:%d, IterCount:%d, LevelCount:%d, DiscK:%f\n", paras.max_d, paras.nIter, paras.nBpLevels, paras.disc_k);

	if(addEdgeInfo == false || (m_iBlockIdx == 0 && m_iBlockIdy == 0)){
		stereoMatchingBP(dataCostPitchedPtr, d_labelImg, d_LabelImgPitch, paras, offsetImg);
		return;
	}

	vector<int> EdgePoints;
	int blockWidth, blockHeight;
	int trueX1, trueY1, trueX2, trueY2;
	m_ImgPartition.GetBlockInfoFull(m_iOffsetX, m_iOffsetY, blockWidth, blockHeight, m_iBlockIdx, m_iBlockIdy, EdgePoints, trueX1, trueY1, trueX2, trueY2);

	int EdgePointCount = blockWidth*2 + blockHeight*2 - 4;
	float * EdgeSmoothInfo = new float[EdgePointCount * paras.max_d];
	memset(EdgeSmoothInfo, 0, EdgePointCount * paras.max_d * sizeof(float));

	int rightBottomX = m_iOffsetX + blockWidth - 1;
	int rightBottomY = m_iOffsetY + blockHeight - 1;

	float Disck = GetTrueDiscK();
	int px, py, x, y;
	for(int PointIndex = 0; PointIndex < EdgePointCount; PointIndex++){
		x = EdgePoints[PointIndex<<1];
		y = EdgePoints[(PointIndex<<1) + 1];
		for(int neighbori = 0; neighbori< NeighborCount; neighbori++)
		{
			px = x + Neighbor[neighbori][0];
			py = y + Neighbor[neighbori][1];

			if(px >= m_iOffsetX && px<= rightBottomX && py >= m_iOffsetY && py <= rightBottomY)
				continue;
			if(px >= 0 && px<ImgWidth && py>=0 && py<ImgHeight && labelImg.at(px, py) != -1){
				for(int di=0; di<paras.max_d; di++)
					EdgeSmoothInfo[PointIndex * paras.max_d + di] += min(abs(di - labelImg.at(px, py)),(int) Disck);
			}
		}
	}

	//add EdgePoint's smooth cost to dataCost
	GpuCpyDataCostDevice2Host(dataCostPitchedPtr, DataCost.GetWidth(), m_iBlockWidth, m_iBlockHeight, paras.max_d,  DataCost.GetBits());
	CDataCostUnit DataCosti;
	for(int PointIndex = 0; PointIndex < EdgePointCount; PointIndex++){
		x = EdgePoints[PointIndex<<1] - m_iOffsetX;
		y = EdgePoints[(PointIndex<<1) + 1] - m_iOffsetY;
		//float * DataCosti = DataCost + (y * blockWidth + x) * m_iMaxD;
		DataCost.GetDataCostUnit(x, y, DataCosti);
		for(int di=0; di<paras.max_d; di++){
			DataCosti[di] += EdgeSmoothInfo[PointIndex * paras.max_d + di];
		}
	}

	//printf("Block Index:(%d,%d), Block Info: offsetX:%d, offsetY:%d, Width:%d, Height:%d\n", m_iBlockIdx, m_iBlockIdy, m_iOffsetX, m_iOffsetY, blockWidth, blockHeight);
	GpuCpyDataCostHost2Device(dataCostPitchedPtr, DataCost.GetWidth(), m_iBlockWidth, m_iBlockHeight, paras.max_d,  DataCost.GetBits());

	stereoMatchingBP(dataCostPitchedPtr, d_labelImg, d_LabelImgPitch, paras, offsetImg);

	//recover the  dataCost
	for(int PointIndex = 0; PointIndex < EdgePointCount; PointIndex++){
		x = EdgePoints[PointIndex<<1] - m_iOffsetX;
		y = EdgePoints[(PointIndex<<1) + 1] - m_iOffsetY;
		DataCost.GetDataCostUnit(x, y, DataCosti);
		for(int di=0; di<paras.max_d; di++){
			DataCosti[di] -= EdgeSmoothInfo[PointIndex * paras.max_d + di];
		}
	}
	GpuCpyDataCostHost2Device(dataCostPitchedPtr, DataCost.GetWidth(), m_iBlockWidth, m_iBlockHeight, paras.max_d,  DataCost.GetBits());
	delete [] EdgeSmoothInfo;
}

//debug
//void CDepthRecoverFuns::RefineDspByGpuBp2(CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr, ZIntImage& labelImg, void* d_labelImg, void* offsetImg, size_t d_LabelImgPitch, bool addEdgeInfo)
//{
//	SBPParas paras;
//	paras.blockWidth = m_iBlockWidth;
//	paras.blockHeight = m_iBlockHeight;
//	paras.disc_k = GetTrueDiscK();
//	paras.max_d = m_iDspLevel;
//	paras.nBpLevels = 5;
//	paras.nIter = 5;
//	paras.offsetX = m_iOffsetX;
//	paras.offsetY = m_iOffsetY;
//	paras.sigma = 0.1F;
//	paras.trueX1 = m_iTrueX1;
//	paras.trueY1 = m_iTrueY1;
//	paras.trueX2 = m_iTrueX2;
//	paras.trueY2 = m_iTrueY2;
//
//	int ImgWidth = labelImg.GetWidth();
//	int ImgHeight = labelImg.GetHeight();
//
//	Shap* shaps = stereoMatchingBP2(dataCostPitchedPtr, d_labelImg, d_LabelImgPitch, paras, offsetImg);
//	CDataCost **datacost = new CDataCost* [paras.nBpLevels];
//
//	int *widths = new int[paras.nBpLevels];
//	int *heights = new int[paras.nBpLevels];
//
//	widths[0] = paras.blockWidth;
//	heights[0] = paras.blockHeight;
//
//	// data pyramid
//	int preWidth = widths[0];
//	int preHeight = heights[0];
//
//	// data costs
//	datacost[0] = &DataCost;
//	//GpuCpyDataCostDevice2Host(dataCostPitchedPtr, datacost[0]->GetWidth(), widths[0], heights[0], m_iDspLevel, datacost[0]->GetBits());
//
//	for(int i=1; i<paras.nBpLevels; i++)
//	{
//		int newWidth = widths[i] = (preWidth + 1) >> 1;
//		int newHeight = heights[i] = (preHeight + 1) >> 1;
//		datacost[i] = new CDataCost(newWidth, newHeight, paras.max_d, true);
//
//		GpuCpyDataCostDevice2Host2(shaps[i].ptr, shaps[i].pitchFloats*sizeof(float), datacost[i]->GetWidth(), widths[i], heights[i], m_iDspLevel, datacost[i]->GetBits());
//
//		preWidth = newWidth;
//		preHeight = newHeight;
//	}
//
//	printf("%d\n", datacost[0]->GetWidth());
//	CBeliefPropagation bp(
//		m_ImgPartition,
//		GetTrueDiscK(),		//disc_k
//		m_iDspLevel ,   //max_d
//		0.1F,			//sigma for gauss smooth
//		5,				//nIter
//		5,				//nlevels
//		m_iCpuThreads	//nThreads
//		);
//	printf("%d\n", datacost[0]->GetWidth());
//	bp.printPara();
//	bp.stereoMatching(m_iBlockIdx, m_iBlockIdy, DataCost, labelImg, addEdgeInfo, datacost);
//
//	return;
//}
//
//
//
//
void CDepthRecoverFuns::RefineDspBySegm( CVDRVideoFrame* currentFrame, ZIntImage& labelImg, CDataCost& DataCost )
{
	ZIntImage segMap(m_iTrueX2 - m_iTrueX1 + 1, m_iTrueY2 - m_iTrueY1 + 1, 1);
	std::vector< std::vector< Wml::Vector2<int> >* > segList;
	currentFrame->Segment(m_iTrueX1, m_iTrueY1, segMap, segList, m_fSegSpatial, m_fSegColor, m_fSegMinsize);
	std::sort(segList.begin(), segList.end(), SizeGreater);

	//for(int iSeg=0; iSeg<segList.size(); ++iSeg){
	//	if(segList[iSeg]->size() >= m_fPlaneFittingSize){
	//		RefineOneSegm(labelImg, segMap, segList[iSeg], DataCost);	//disc_k);
	//	}
	//}

	CParallelManager pm(m_iCpuThreads);
	for(int iSeg=0; iSeg<segList.size(); ++iSeg){
		if(segList[iSeg]->size() >= m_fPlaneFittingSize){
			CPlanFittingWorkUnit* pWorkUnit = new CPlanFittingWorkUnit(labelImg, segMap, segList[iSeg], DataCost,currentFrame);
			pm.EnQueue(pWorkUnit);
		}
	}
	pm.Run();
	//Clear Segmentation
	for(int i=0; i<segList.size(); ++i){
		delete segList[i];
	}
}

void CDepthRecoverFuns::RefineOneSegm( ZIntImage& labelImg, ZIntImage& segMap, std::vector< Wml::Vector2<int> >* pSegmPoints,  CDataCost& dataCost,CVDRVideoFrame* pCurrentFrame)
{
	//SaveZimg(m_skyClassifier,"he1");
	if(m_bUseSC){
	int skyclassifyEnergy=0;
	for(std::vector< Wml::Vector2<int> >::iterator ptIter = pSegmPoints->begin(); ptIter != pSegmPoints->end(); ++ptIter){
		int x = ptIter->X();
		int y = ptIter->Y();
	//	printf("%d  %d  %d\n",x,y,m_skyClassifier.at(x,y));
		if(m_skyClassifier.at(m_iTrueX1 + x, m_iTrueY1 + y) !=0) skyclassifyEnergy++;
	}
//	printf("%d\n",skyclassifyEnergy);
	if(skyclassifyEnergy > pSegmPoints->size()*0.7){//为天空，赋予很大的深度值。
		for(std::vector< Wml::Vector2<int> >::iterator ptIter = pSegmPoints->begin(); ptIter != pSegmPoints->end(); ++ptIter){
			int x = ptIter->X();
			int y = ptIter->Y();
			pCurrentFrame->SetDspAt(m_iTrueX1 +x,m_iTrueY1 +  y, GetDspAtLevelI(0));
			labelImg.at(m_iTrueX1 + x, m_iTrueY1 + y) = 0;
		}
		printf("Sky segm!\n");
		return;
	}
	}

	int iWidth = segMap.GetWidth();
	int iHeight = segMap.GetHeight();

	int offsetX = m_iTrueX1 - m_iOffsetX;
	int offsetY = m_iTrueY1 - m_iOffsetY;

	int dspLevel = dataCost.GetChannel();
	std::vector<double> layerEnergy;
	layerEnergy.resize(dspLevel);
	double minSegEnergy = 1e20;
	double minPriorEnergy=1e20;
	int iSegBestLayer=0;
	int iPriorBestLayer=0;
	float Disck = GetTrueDiscK();
	Wml::Vector4d PlanA;
	PlanA[2]=0;
	double dp=0.0;
	if(m_bUsePriorPlane)
	{
	for(int i=0;i<m_iPlaneNum;i++)//计算minEnergy和bestPriorLabel
	{
		for(std::vector< Wml::Vector2<int> >::iterator it = pSegmPoints->begin(); it != pSegmPoints->end(); ++it){
			int x = it->X();
			int y = it->Y();
			//layerEnergy[iLayer] += dataCost[((offsetX + x) + (offsetY + y) * m_iBlockWidth) * dspLevel + iLayer];
			double dsp=0.0;
			pCurrentFrame->GetDpFrmImgCoordAndPlan(m_iTrueX1+x,m_iTrueY1+y,m_PlanArr[i],dsp);
			dsp=1/dsp;
			dsp=max(dsp,m_dDspMin);
			dsp=min(dsp,m_dDspMax);
			layerEnergy[i] += dataCost.GetValueAt(offsetX+x, offsetY+y, (int )((dsp-m_dDspMin)/(m_dDspMax-m_dDspMin)*(dspLevel-1)));//每一层的Energy等于segm中每一个点在该层的datacost之和。
			//Neighbor
// 			int x1,y1;
// 			for(int neighbori = 0; neighbori< NeighborCount; neighbori++){
// 				x1 = x + Neighbor[neighbori][0];
// 				y1 = y + Neighbor[neighbori][1];
// 				if(x1>=0 && x1<iWidth && y1>=0 && y1<iHeight && segMap.at(x,y) != segMap.at(x1,y1) && labelImg.at(x1,y1) != -1){
// 					layerEnergy[i] += min(Disck, (float)abs( i - labelImg.at(x1, y1)));
// 				}
// 			}
		}
		if(minPriorEnergy > layerEnergy[i]){
			minPriorEnergy = layerEnergy[i];
			iPriorBestLayer = i;
		}
	}
 	for(int i=0;i<m_iPlaneNum;i++)
 	{
 		for(std::vector< Wml::Vector2<int> >::iterator it = pSegmPoints->begin(); it != pSegmPoints->end(); ++it){
 			int x =m_iTrueX1+it->X();
 			int y =m_iTrueY1+it->Y();
 			//layerEnergy[iLayer] += dataCost[((offsetX + x) + (offsetY + y) * m_iBlockWidth) * dspLevel + iLayer];
 			double dp=0.0;
 			pCurrentFrame->GetDpFrmImgCoordAndPlan(x,y,m_PlanArr[i],dp);
 			dp=1/dp;
 			dp=max(dp,m_dDspMin);
 			dp=min(dp,m_dDspMax);
 			//dp=(dp-dspMin)/(dspMax-dspMin)*(dspLevel-1);
 			//	printf("Dp=%d  ",(int)((dp-dspMin)/(dspMax-dspMin)*(dspLevel-1)));
 			segm[i].at(x,y)=(int)((dp-m_dDspMin)/(m_dDspMax-m_dDspMin)*(dspLevel-1));
 		}
 	}

	Wml::Vector3d XX,YY,ZZ;
	for(int idx=0;idx<3;idx++)
	{
		std::vector< Wml::Vector2<int> >::iterator it = pSegmPoints->begin()+(idx+1)*(pSegmPoints->end()-pSegmPoints->begin()-1)/5;

		XX[idx]=m_iTrueX1+it->X();
		YY[idx]=m_iTrueY1+it->Y();
		pCurrentFrame->GetDpFrmImgCoordAndPlan(m_iTrueX1+it->X(),m_iTrueY1+it->Y(),m_PlanArr[iPriorBestLayer],dp);
		dp=1/dp;
		ZZ[idx]=((dp-m_dDspMin)/(m_dDspMax-m_dDspMin)*(dspLevel-1));
	}

	PlanA[0]=(YY[0]-YY[1])*(ZZ[0]-ZZ[2])-(ZZ[0]-ZZ[1])*(YY[0]-YY[2]);
	PlanA[1]=(ZZ[0]-ZZ[1])*(XX[0]-XX[2])-(XX[0]-XX[1])*(ZZ[0]-ZZ[2]);
	PlanA[2]=(XX[0]-XX[1])*(YY[0]-YY[2])-(YY[0]-YY[1])*(XX[0]-XX[2]);
	PlanA[3]=-(PlanA[0]*XX[0]+PlanA[1]*YY[0]+PlanA[2]*ZZ[0]);
	for(std::vector< Wml::Vector2<int> >::iterator it = pSegmPoints->begin(); it != pSegmPoints->end(); ++it){
		int x = it->X();
		int y = it->Y();
		//layerEnergy[iLayer] += dataCost[((offsetX + x) + (offsetY + y) * m_iBlockWidth) * dspLevel + iLayer];
	//	double dp=0.0;
		pCurrentFrame->GetDpFrmImgCoordAndPlan(m_iTrueX1+it->X(),m_iTrueY1+it->Y(),m_PlanArr[iPriorBestLayer],dp);
		dp=1/dp;
		double ta,tb,tc;
		if(fabs(PlanA[2])<=1e-20 ) {ta=0;tb=0;tc=0;}
		else
		{
			ta = -PlanA[0]/PlanA[2];
			tb = -PlanA[1]/PlanA[2];
			tc = -PlanA[3]/PlanA[2];
		}
	//	dp=ta*(x+m_iTrueX1)+tb*(y+m_iTrueY1)+tc;
		dp=((dp-m_dDspMin)/(m_dDspMax-m_dDspMin)*(dspLevel-1));
		m_segm.at(x+m_iTrueX1,  y+m_iTrueY1)=(int)min( (double)dspLevel-1, max(0.0, dp+0.5) );
	}
	}

	//for(int i=0;i<m_iPlaneNum;i++)
	//{
	//	for(std::vector< Wml::Vector2<int> >::iterator it = pSegmPoints->begin(); it != pSegmPoints->end(); ++it){
	//		int x = m_iTrueX1+it->X();
	//		int y = m_iTrueY1+it->Y();
	//		//layerEnergy[iLayer] += dataCost[((offsetX + x) + (offsetY + y) * m_iBlockWidth) * dspLevel + iLayer];
	//		double dp=0.0;
	//		pCurrentFrame->GetDpFrmImgCoordAndPlan(x,y,m_PlanArr[i],dp);
	//		dp=1/dp;
	//		dp=max(dp,m_dDspMin);
	//		dp=min(dp,m_dDspMax);
	//		//dp=(dp-dspMin)/(dspMax-dspMin)*(dspLevel-1);
	//		//	printf("Dp=%d  ",(int)((dp-dspMin)/(dspMax-dspMin)*(dspLevel-1)));
	//		segm[i].at(x,y)=(int)((dp-m_dDspMin)/(m_dDspMax-m_dDspMin)*(dspLevel-1));
	//	}
	//}
	for(int iLayer=0; iLayer<dspLevel; ++iLayer){
		layerEnergy[iLayer] = 0.0;
		for(std::vector< Wml::Vector2<int> >::iterator ptIter = pSegmPoints->begin(); ptIter != pSegmPoints->end(); ++ptIter){
			int x = ptIter->X();
			int y = ptIter->Y();

			int dis = ptIter - pSegmPoints->begin();

			//layerEnergy[iLayer] += dataCost[((offsetX + x) + (offsetY + y) * m_iBlockWidth) * dspLevel + iLayer];
			layerEnergy[iLayer] += dataCost.GetValueAt(offsetX+x, offsetY+y, iLayer);

			//Neighbor
			int x1,y1;
			for(int neighbori = 0; neighbori< NeighborCount; neighbori++){
				x1 = x + Neighbor[neighbori][0];
				y1 = y + Neighbor[neighbori][1];
				if(x1>=0 && x1<iWidth && y1>=0 && y1<iHeight && segMap.at(x,y) != segMap.at(x1,y1) ){
					layerEnergy[iLayer] += min(Disck,(float)(abs( iLayer - labelImg.at(m_iTrueX1 + x1, m_iTrueY1 + y1))));
				}
			}
		}
		if(minSegEnergy > layerEnergy[iLayer]){
			minSegEnergy = layerEnergy[iLayer];
			iSegBestLayer = iLayer;
		}
	}

	int segPointCount = pSegmPoints->size();
	CLevenMarOptimizer LMOptimizer(*pSegmPoints, 25, segPointCount, dspLevel, Disck);

	double nonSegEnergy = 0;
	double nonSegSmoothCost = 0;
	//double nonSegDataCost = 0;//test
	CDataCostUnit dataCostUnit;
	for(int index = 0; index < segPointCount; index ++){
		int x = (*pSegmPoints)[index].X();
		int y = (*pSegmPoints)[index].Y();

		int iBestLayer = labelImg.at(m_iTrueX1 + x, m_iTrueY1 + y);

		int x1, y1;
		for(int neighbori = 0; neighbori< NeighborCount; neighbori++){
			x1 = x + Neighbor[neighbori][0];
			y1 = y + Neighbor[neighbori][1];
			if(x1<0 || x1>=iWidth || y1<0 || y1>=iHeight)
				continue;
			if(segMap.at(x,y) != segMap.at(x1,y1) ){
				LMOptimizer.AddEdgeNeighborInfo(index, labelImg.at(m_iTrueX1 + x1, m_iTrueY1 + y1));
				nonSegEnergy += min(Disck, (float)(abs( iBestLayer - (int)labelImg.at(m_iTrueX1 + x1, m_iTrueY1 + y1))));
				//nonSegDataCost += min(Disck, abs( iBestLayer - labelImg.at(x1,y1)));
			}
			else
				nonSegSmoothCost += min(Disck, (float)(abs( iBestLayer - (int)labelImg.at(m_iTrueX1 + x1, m_iTrueY1 + y1))));
		}
		dataCost.GetDataCostUnit(offsetX + x, offsetY + y, dataCostUnit);
		//LMOptimizer.SetDataCostAt(index, &dataCost[((offsetX + x) + (offsetY + y) * m_iBlockWidth) * dspLevel]);
		//nonSegEnergy += dataCost[((offsetX + x) + (offsetY + y) * m_iBlockWidth) * dspLevel + iBestLayer];
		LMOptimizer.SetDataCostAt(index, dataCostUnit);
		iBestLayer=max(iBestLayer,0);
		nonSegEnergy += dataCost.GetValueAt(offsetX+x, offsetY+y, iBestLayer);
	}
	nonSegEnergy += nonSegSmoothCost/2.0;
	//printf("nonSegDataCost:%lf nonSegSoothCost:%lf\n", nonSegDataCost, nonSegEnergy-nonSegDataCost);

	double MinValue;
	Wml::GVectord optA(3);
	double initA,initB,initC;
 //	if(m_bUsePriorPlane){
// 	if(fabs(PlanA[2])<=1e-20) {optA[0]=0;optA[1]=0;optA[2]=0;}
// 	else{
// 		optA[0] = -PlanA[0]/PlanA[2];
// 		optA[1] = -PlanA[1]/PlanA[2];
// 		optA[2] = -PlanA[3]/PlanA[2];
// 	}
// 	initA=optA[0];
// 	initB=optA[1];
// 	initC=optA[2];
//  	}
//  	else{
		optA[0] = 0.0; optA[1] = 0.0; optA[2] = iSegBestLayer;
//	}

	LMOptimizer.Optimize(optA, MinValue);

	double a,b,c;
	a = optA[0]; b = optA[1];	c = optA[2];

	//printf("iBestLayer:%d\n",iSegBestLayer);
	//printf("a:%f,b:%f,c:%f\n",a,b,c);

	//Valid it again!
	minSegEnergy = LMOptimizer.GetAbsValue(optA);

	//printf("minSegEnergy:%lf, nonSegEnergy:%lf\n",minSegEnergy,nonSegEnergy);
	//exit(0);
	double segErrRateThreshold=GetTruetSegErrRateThreshold();
	if(( minSegEnergy < segErrRateThreshold * nonSegEnergy && !m_bUsePriorPlane ) || (minSegEnergy < segErrRateThreshold * nonSegEnergy && m_bUsePriorPlane && minSegEnergy <  minPriorEnergy)){
		for(std::vector< Wml::Vector2<int> >::iterator ptIter = pSegmPoints->begin(); ptIter != pSegmPoints->end(); ++ptIter){
			int x = ptIter->X();
			int y = ptIter->Y();
			float bestD = a * x + b * y + c;
			double xx=GetDspAtLevelI(bestD);
			pCurrentFrame->SetDspAt(m_iTrueX1 +x,m_iTrueY1 +  y, GetDspAtLevelI(bestD));
			labelImg.at(m_iTrueX1 + x, m_iTrueY1 + y) = min( dspLevel-1, (int)(max(0.0,bestD+0.5)));
		}
	}
	else if(m_bUsePriorPlane && minSegEnergy >=  minPriorEnergy && minPriorEnergy <1.5 *  nonSegEnergy)
	{
		for(std::vector< Wml::Vector2<int> >::iterator ptIter = pSegmPoints->begin(); ptIter != pSegmPoints->end(); ++ptIter){
			int x = ptIter->X();
			int y = ptIter->Y();
			pCurrentFrame->GetDpFrmImgCoordAndPlan(m_iTrueX1+ptIter->X(),m_iTrueY1+ptIter->Y(),m_PlanArr[iPriorBestLayer],dp);
			dp=1/dp;
			pCurrentFrame->SetDspAt(m_iTrueX1 +x, m_iTrueY1 +  y, dp);
			dp=((dp-m_dDspMin)/(m_dDspMax-m_dDspMin)*(dspLevel-1));
			labelImg.at(m_iTrueX1 +x, m_iTrueY1 +  y) =min( (double)dspLevel-1, max(0.0, dp+0.5) );
		}
// 		a=initA;
// 		b=initB;
// 		c=initC;
// 		for(std::vector< Wml::Vector2<int> >::iterator ptIter = pSegmPoints->begin(); ptIter != pSegmPoints->end(); ++ptIter){
// 			int x = ptIter->X();
// 			int y = ptIter->Y();
// 			float bestD = a * (x+m_iTrueX1) + b * (y+m_iTrueY1) + c;
// 			pCurrentFrame->SetDspAt(m_iTrueX1 +x, m_iTrueY1 +  y, GetDspAtLevelI(bestD));
// 			labelImg.at(m_iTrueX1 +x, m_iTrueY1 +  y) =min( (double)dspLevel-1, max(0.0, bestD+0.5) );
// 		}
	}
		}

void CDepthRecoverFuns::EstimateDepth( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost)
{
	if(m_bUseSC){
	m_skyClassifier.CreateAndInit(CVDRVideoFrame::GetImgWidth(),CVDRVideoFrame::GetImgHeight(),1,0);
	TrainingDataGeneratorAndValidator TDG;
	TDG.ValidateOnImageUsingMRF(1, 0, 0, *m_RF,pCurrentFrame->m_sImgPathName,m_skyClassifier);
	SaveZimg(m_skyClassifier,"skyclassifyResult");
	}
	if(m_bUsePriorPlane) m_segm.CreateAndInit(CVDRVideoFrame::GetImgWidth(),CVDRVideoFrame::GetImgHeight(),1,0);
	ZIntImage labelImg;
	labelImg.CreateAndInit( CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), 1, -1);

 	for(int idx=0;idx<11;idx++)
 	segm[idx].CreateAndInit( CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), 1, -1);

	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());

	std::vector<Wml::Matrix4d> PList(nearFrames.size());
	GetPList(PList, pCurrentFrame, nearFrames);

	int dspLevel = DataCost.GetChannel();

	//load
	pCurrentFrame->LoadColorImg();
	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->LoadColorImg();
	int blockCountX = m_ImgPartition.getBlockCountX();
	int blockCountY = m_ImgPartition.getBlockCountY();

	for(int blockIdy = 0; blockIdy < blockCountY; blockIdy++)
		for(int blockIdX = 0; blockIdX < blockCountX; blockIdX++){
			int offsetX, offsetY;
			int blockWidth, blockHeight;
			int trueX1, trueY1, trueX2, trueY2;
			m_ImgPartition.GetBlockInfoSimple(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			SetBlockState(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);

			clock_t tempTime = clock();
			printf("Init DataCost:\n");
			printf("DataCostWeight: %f\n",GetTrueDataCostWeight());
			PrintBlockState();
			GetInitialDataCost(nearFrames, PList, pCurrentFrame, DataCost, labelImg);

			//pCurrentFrame->SaveDataCost(DataCost, m_iBlockWidth * m_iBlockHeight * dspLevel * sizeof(float));
			//pCurrentFrame->ReadDataCost(DataCost, labelImg, m_iOffsetX, m_iOffsetY, m_iBlockWidth, m_iBlockHeight );
			//if(m_ifGenerateTempResult == true)
			//	pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::INIT, dspLevel);
// 			for(int tx=0;tx<blockWidth;tx++)
// 				for(int ty=0;ty<blockHeight;ty++)
// 					for()
// 					{
// 						DataCost.At(tx,ty,td);
// 					}

			printf("\nTime:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

			printf("Init By BP:\n");
			RefineDspByBP(DataCost, labelImg, true);
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();
			pCurrentFrame->SetDspImg(labelImg, trueX1, trueY1, trueX2, trueY2);
			if(m_ifGenerateTempResult == true)
				pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);
			printf("Init By Segmentation:\n");

			PrintBlockState();
			CDepthRecoverFuns::GetInstance()->RefineDspBySegm(pCurrentFrame, labelImg, DataCost);
			//if(m_ifGenerateTempResult == true)
			//	pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::DeptExpansion, dspLevel);
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);

//
	}
    if(m_bUsePriorPlane && m_ifGenerateTempResult == true && m_ifGenerateTempResult == true)
		SaveZimg(m_segm,"prior_planes_result");
	char filename[20];
	for(int idx=0;idx<11;idx++){
		sprintf(filename,"Plane%d",idx);
		SaveZimg(segm[idx],filename);}
	if(m_ifGenerateTempResult == true){
	//	pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::INIT, dspLevel);
		pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::INIT,m_dDspMin,m_dDspMax);
	//	pCurrentFrame->InitLabelImgByDspImg(labelImg, dspLevel);
	//	pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::BO, dspLevel);
	}

	pCurrentFrame->SaveDspImg();
	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->Clear();
	pCurrentFrame->Clear();
}

void CDepthRecoverFuns::EstimateDepthByGpu( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr )
{
	if(m_bUseSC){
	m_skyClassifier.CreateAndInit(CVDRVideoFrame::GetImgWidth(),CVDRVideoFrame::GetImgHeight(),1,0);
	TrainingDataGeneratorAndValidator TDG;
	TDG.ValidateOnImageUsingMRF(1, 0, 0, *m_RF,pCurrentFrame->m_sImgPathName,m_skyClassifier);
	SaveZimg(m_skyClassifier,"skyclassifyResult");
    }
	if(m_bUsePriorPlane) m_segm.CreateAndInit(CVDRVideoFrame::GetImgWidth(),CVDRVideoFrame::GetImgHeight(),1,0);
	int ImgHeight = CVDRVideoFrame::GetImgHeight();
	int ImgWidth = CVDRVideoFrame::GetImgWidth();

	ZIntImage labelImg;
	labelImg.CreateAndInit( CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), 1, -1);

	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());

	int dspLevel = DataCost.GetChannel();

	//load
	pCurrentFrame->LoadColorImg();
	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->LoadColorImg();

	pCurrentFrame->SetDspImg(labelImg,0,0,ImgWidth-1,ImgHeight-1);

	//gpu
	size_t ColorImgPitchBytes;
	int count = nearFrames.size() + 1;
	int * d_labelImg;
	size_t d_labeImgPitch;

	//color image
	unsigned char **colorImgs = new unsigned char *[count];
	colorImgs[0] = (unsigned char *)pCurrentFrame->m_pColorImg->GetMap();
	for(int i=1; i<count; i++)
		colorImgs[i] = (unsigned char *)nearFrames[i-1]->m_pColorImg->GetMap();
	int EffectWidth = pCurrentFrame->m_pColorImg->GetEffectWidth();
	GpuSetColorImg(colorImgs, EffectWidth, ImgWidth, ImgHeight, count, &ColorImgPitchBytes);
	delete [] colorImgs;

	//camera parameters
	float* cameraParameters = new float[count * 21];
	pCurrentFrame->GetCameraParas(&(cameraParameters[0]));
	for(int i=1; i<count; i++)
		nearFrames[i-1]->GetCameraParas(&(cameraParameters[i * 21]));
	GpuSetCameraParas(cameraParameters, count);
	delete[] cameraParameters;

	//dspV
	GpuSetDspV(&m_fDspV[0], dspLevel);

	//labelImg
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_labelImg), &d_labeImgPitch);

	int blockCountX = m_ImgPartition.getBlockCountX();
	int blockCountY = m_ImgPartition.getBlockCountY();

	for(int blockIdy = 0; blockIdy < blockCountY; blockIdy++)
		for(int blockIdX = 0; blockIdX < blockCountX; blockIdX++){
			int offsetX, offsetY;
			int blockWidth, blockHeight;
			int trueX1, trueY1, trueX2, trueY2;
			m_ImgPartition.GetBlockInfoSimple(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			SetBlockState(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			PrintBlockState();

			clock_t tempTime = clock();
			printf("Init DataCost:\n");

			SDataCostParas pars;
			pars.blockHeight = blockHeight;
			pars.blockWidth = blockWidth;
			pars.dspLevels = dspLevel;
			pars.ImgHeight = ImgHeight;
			pars.ImgWidth  = ImgWidth;
			pars.m_dColorSigma = m_dColorSigma;
			pars.m_fColorMissPenalty = m_fColorMissPenalty;
			pars.m_fDataCostWeight = GetTrueDataCostWeight();
			pars.offsetX = offsetX;
			pars.offsetY = offsetY;
			pars.TrueX1 = m_iTrueX1;
			pars.TrueY1 = m_iTrueY1;
			pars.TrueX2 = m_iTrueX2;
			pars.TrueY2 = m_iTrueY2;
			GpuGetInitDataCost(count, ColorImgPitchBytes, pars, DataCost.GetBits(), DataCost.GetWidth(), dataCostPitchedPtr, d_labelImg, d_labeImgPitch);
			//GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
			//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);

			//pCurrentFrame->SaveDataCost(DataCost, m_iBlockWidth * m_iBlockHeight * dspLevel * sizeof(float));
			//pCurrentFrame->ReadDataCost(DataCost, labelImg, m_iOffsetX, m_iOffsetY, m_iBlockWidth, m_iBlockHeight );
			//if(m_ifGenerateTempResult == true)
				//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");

			//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
			//pCurrentFrame->SaveLabelImg(labelImg,  CVDRVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");

			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();
			//DataCost.SetLabelImg(m_iOffsetX, m_iOffsetY, m_iBlockWidth, m_iBlockHeight, labelImg);

			printf("Init By BP:\n");
			//RefineDspByBP(DataCost, labelImg);
			RefineDspByGpuBp(DataCost, dataCostPitchedPtr, labelImg, d_labelImg, NULL, d_labeImgPitch, true);
			GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
			//if(m_ifGenerateTempResult == true)
			//	pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::BO, dspLevel);

			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();
			//exit(0);
			pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
			printf("Init By Segmentation:\n");
			//PrintBlockState();
			CDepthRecoverFuns::GetInstance()->RefineDspBySegm(pCurrentFrame, labelImg, DataCost);
			//if(m_ifGenerateTempResult == true)
			//	pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::DeptExpansion, dspLevel);
			printf("Time:%.3f s\n\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
// 			ZIntImage *depthimage=new ZIntImage(m_iTrueX2-m_iTrueX1+1,m_iTrueY2-trueY1+1,1);
// 			//depthimage->CreateAndInit(m_iTrueX2-m_iTrueX1+1,m_iTrueY2-trueY1+1,1);
// 			for(int ii=m_iTrueX1;ii<m_iTrueX2;ii++)
// 				for(int jj=m_iTrueY1;jj<m_iTrueY2;jj++)
// 					depthimage->SetPixel(ii,jj,0,GetLevelFromDspi(pCurrentFrame->GetDspAt(ii,jj),dspLevel) );
	//		SaveZimg(*depthimage,"result");
//
		}

		GpuClearColorImg(count);
		GpuDeleMem(d_labelImg);
		if(m_bUsePriorPlane && m_ifGenerateTempResult == true && m_ifGenerateTempResult == true)
			SaveZimg(m_segm,"prior_planes_result");
		if(m_ifGenerateTempResult == true){
// 			pCurrentFrame->InitLabelImgByDspImg(labelImg, dspLevel);
// 			pCurrentFrame->SaveLabelImg(labelImg,CVDRVideoFrame::BO,dspLevel);
			pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::INIT,m_dDspMin,m_dDspMax);
		}

		pCurrentFrame->SaveDspImg();
		for(int i=0; i<nearFrames.size(); i++)
			nearFrames[i]->Clear();
		pCurrentFrame->Clear();
}

void CDepthRecoverFuns::RefineDepth( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost)
{
	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());
	//load
	pCurrentFrame->LoadColorImg();
	pCurrentFrame->LoadDepthImg();
	for(int i=0; i<nearFrames.size(); i++){
		nearFrames[i]->LoadDepthImg();
		nearFrames[i]->LoadColorImg();
	}

	int dspLevel = DataCost.GetChannel();

	ZIntImage labelImg, tmpLableImg, offsetImg;
	pCurrentFrame->InitLabelImgByDspImg(labelImg, dspLevel);
	offsetImg.CreateAndInit(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), 1, -1);
	//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::DeptExpansion, dspLevel);

	std::vector<Wml::Matrix4d> PList(nearFrames.size());
	GetPList(PList, pCurrentFrame, nearFrames);

	int blockCountX = m_ImgPartition.getBlockCountX();
	int blockCountY = m_ImgPartition.getBlockCountY();
	for(int blockIdy = 0; blockIdy < blockCountY; blockIdy++)
		for(int blockIdX = 0; blockIdX < blockCountX; blockIdX++){
			printf("\nRun Bo!\n");
			printf("DataCostWeight: %f\n",GetTrueDataCostWeight());
			int offsetX, offsetY;
			int blockWidth, blockHeight;
			int trueX1, trueY1, trueX2, trueY2;
			m_ImgPartition.GetBlockInfoSimple(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			SetBlockState(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);

			clock_t tempTime =  clock();
			printf("Refine DataCost:\n");
			PrintBlockState();
			GetBundleDataCost(nearFrames, PList, pCurrentFrame, DataCost, labelImg);

			//pCurrentFrame->SaveDataCost(DataCost);
			//pCurrentFrame->ReadDataCost(DataCost, labelImg);
			//pCurrentFrame->SetDspImg(labelImg);
			//if(m_ifGenerateTempResult == true){
				//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
				//system("pause");

				//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
				//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
				//system("pause");
			//}
			printf("\nTime:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

//			pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::INIT, dspLevel);

			printf("Refine By BP:\n");
			RefineDspByBP(DataCost, labelImg, true);
			pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2 );
			//if(m_ifGenerateTempResult == true){
			//	pCurrentFrame->SaveLabelImg(labelImg,  CVDRVideoFrame::TYPE::DeptExpansion, dspLevel);
			//	//system("pause");
			//}
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

			if( m_iSubSample > 1){
				printf("Refine By Depth Expansion:\n");
				PrintBlockState();
				tmpLableImg = labelImg;
				_SuperResolutionRefine(pCurrentFrame, nearFrames, 1, PList, offsetImg, tmpLableImg);
				int newdspLevel = (dspLevel - 1) * m_iSubSample + 1;
				pCurrentFrame->SetDspImg(tmpLableImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2, m_dDspMin, m_dDspMax, newdspLevel);

				//if(m_ifGenerateTempResult == true){
					//pCurrentFrame->SaveDspExpanLabelImg(tmpLableImg, dspLevel, m_iOffsetX, m_iOffsetY, m_iBlockWidth, m_iBlockHeight, (dspLevel - 1) * m_iSubSample + 1);
				//}
				//if(m_ifGenerateTempResult == true){
				//	pCurrentFrame->SaveLabelImg(tmpLableImg,  CVideoFrame::TYPE::INIT, (dspLevel - 1) * m_iSubSample + 1);
				//	system("pause");
				//}
				printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			}
		}

	if(m_ifGenerateTempResult == true){
		if( m_iSubSample > 1){
			int newdspLevel = (dspLevel - 1) * m_iSubSample + 1;
			//pCurrentFrame->InitLabelImgByDspImg(labelImg, newdspLevel);
			pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);;
		}
		else
			pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);
	}

	pCurrentFrame->SaveDspImg();
	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->Clear();
	pCurrentFrame->Clear();
}

void CDepthRecoverFuns::RefineDepthPF( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost)
{
	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());
	//load
	pCurrentFrame->LoadColorImg();
	pCurrentFrame->LoadDepthImg();
	for(int i=0; i<nearFrames.size(); i++){
		nearFrames[i]->LoadDepthImg();
		nearFrames[i]->LoadColorImg();
	}

	int dspLevel = DataCost.GetChannel();

	ZIntImage labelImg, tmpLableImg, offsetImg;
	pCurrentFrame->InitLabelImgByDspImg(labelImg, dspLevel);
	offsetImg.CreateAndInit(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), 1, -1);
	//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::DeptExpansion, dspLevel);

	std::vector<Wml::Matrix4d> PList(nearFrames.size());
	GetPList(PList, pCurrentFrame, nearFrames);

	int blockCountX = m_ImgPartition.getBlockCountX();
	int blockCountY = m_ImgPartition.getBlockCountY();
	for(int blockIdy = 0; blockIdy < blockCountY; blockIdy++)
		for(int blockIdX = 0; blockIdX < blockCountX; blockIdX++){
			printf("\nBo With PlanFitting!\n");
			int offsetX, offsetY;
			int blockWidth, blockHeight;
			int trueX1, trueY1, trueX2, trueY2;
			m_ImgPartition.GetBlockInfoSimple(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			SetBlockState(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);

			clock_t tempTime =  clock();
			printf("Refine DataCost:\n");
			PrintBlockState();
			GetBundleDataCost(nearFrames, PList, pCurrentFrame, DataCost, labelImg);

			//pCurrentFrame->SaveDataCost(DataCost);
			//pCurrentFrame->ReadDataCost(DataCost, labelImg);
			//pCurrentFrame->SetDspImg(labelImg);
			//if(m_ifGenerateTempResult == true){
			//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");

			//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
			//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");
			//}
			printf("\nTime:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

			printf("Refine By BP:\n");
			RefineDspByBP(DataCost, labelImg, true);
			pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2 );
			//if(m_ifGenerateTempResult == true){
			//	pCurrentFrame->SaveLabelImg(labelImg,  CVDRVideoFrame::TYPE::BO, dspLevel);
			//	//system("pause");
			//}
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

			printf("Refine By PlanFitting:\n");
			PrintBlockState();

		//	CDepthRecoverFuns::GetInstance()->SetSegErrRateThreshold(15.0);

			CDepthRecoverFuns::GetInstance()->RefineDspBySegm(pCurrentFrame, labelImg, DataCost);

			if( m_iSubSample > 1){
				printf("Refine By Depth Expansion:\n");
				PrintBlockState();
				tmpLableImg = labelImg;
				_SuperResolutionRefine(pCurrentFrame, nearFrames, 1, PList, offsetImg, tmpLableImg);
				int newdspLevel = (dspLevel - 1) * m_iSubSample + 1;
				pCurrentFrame->SetDspImg(tmpLableImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2, m_dDspMin, m_dDspMax, newdspLevel);

				//if(m_ifGenerateTempResult == true){
				//pCurrentFrame->SaveDspExpanLabelImg(tmpLableImg, dspLevel, m_iOffsetX, m_iOffsetY, m_iBlockWidth, m_iBlockHeight, (dspLevel - 1) * m_iSubSample + 1);
				//}
				//if(m_ifGenerateTempResult == true){
				//	pCurrentFrame->SaveLabelImg(tmpLableImg,  CVideoFrame::TYPE::INIT, (dspLevel - 1) * m_iSubSample + 1);
				//	system("pause");
				//}
				printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			}
		}

		if(m_ifGenerateTempResult == true){
			if( m_iSubSample > 1){
				int newdspLevel = (dspLevel - 1) * m_iSubSample + 1;
				//pCurrentFrame->InitLabelImgByDspImg(labelImg, newdspLevel);
				pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);
			}
			else
				pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);
		}

		pCurrentFrame->SaveDspImg();
		for(int i=0; i<nearFrames.size(); i++)
			nearFrames[i]->Clear();
		pCurrentFrame->Clear();
}

void CDepthRecoverFuns::RefineDepthByGpu( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame,
										 CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr, CDataCost& subDataCost, cudaPitchedPtr& subDataCostPitchPtr)
{
	int ImgHeight = CVDRVideoFrame::GetImgHeight();
	int ImgWidth = CVDRVideoFrame::GetImgWidth();

	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());
	//load
	pCurrentFrame->LoadColorImg();
	pCurrentFrame->LoadDepthImg();
	for(int i=0; i<nearFrames.size(); i++){
		nearFrames[i]->LoadDepthImg();
		nearFrames[i]->LoadColorImg();
	}

	int dspLevel = DataCost.GetChannel();

	ZIntImage labelImg;
	pCurrentFrame->InitLabelImgByDspImg(labelImg, dspLevel);
	//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::DeptExpansion, dspLevel);

	int count = nearFrames.size() + 1;
	//gpu
	size_t ColorImgPitchBytes;
	size_t DspImgPitchBytes;
	int * d_labelImg;
	int * d_tmpLableImg;
	int * d_offsetImg;
	size_t d_labeImgPitch;

	//color image
	unsigned char **colorImgs = new unsigned char *[count];
	colorImgs[0] = (unsigned char *)pCurrentFrame->m_pColorImg->GetMap();
	for(int i=1; i<count; i++)
		colorImgs[i] = (unsigned char *)nearFrames[i-1]->m_pColorImg->GetMap();
	int EffectWidthBytes = pCurrentFrame->m_pColorImg->GetEffectWidth();
	GpuSetColorImg(colorImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, &ColorImgPitchBytes);
	delete [] colorImgs;

	//Dsp image
	float **dspImgs = new float *[count];
	dspImgs[0] = (float *)pCurrentFrame->m_pDspImg->GetMap();
	for(int i=1; i<count; i++)
		dspImgs[i] = (float *)nearFrames[i-1]->m_pDspImg->GetMap();
	EffectWidthBytes = pCurrentFrame->m_pDspImg->GetEffectWidth();
	GpuSetDspImg(dspImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, &DspImgPitchBytes);
	//GpuGetDspImg(dspImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, DspImgPitchBytes);
	delete [] dspImgs;

	//camera parameters
	float* cameraParameters = new float[count * 21];
	pCurrentFrame->GetCameraParas(&(cameraParameters[0]));
	for(int i=1; i<count; i++)
		nearFrames[i-1]->GetCameraParas(&(cameraParameters[i * 21]));
	GpuSetCameraParas(cameraParameters, count);
	delete[] cameraParameters;

	//dspV
	GpuSetDspV(&m_fDspV[0], dspLevel);

	//labelImg
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_tmpLableImg), &d_labeImgPitch);
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_labelImg), &d_labeImgPitch);
	GpuSetGpuLabelImgByLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, 0, 0, ImgWidth-1, ImgHeight - 1);

	//offsetImg
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_offsetImg), &d_labeImgPitch);

	//debug test
	//std::vector<Wml::Matrix4d> PList(nearFrames.size());
	//GetPList(PList, pCurrentFrame, nearFrames);

	int blockCountX = m_ImgPartition.getBlockCountX();
	int blockCountY = m_ImgPartition.getBlockCountY();
	for(int blockIdy = 0; blockIdy < blockCountY; blockIdy++)
		for(int blockIdX = 0; blockIdX < blockCountX; blockIdX++){
			int offsetX, offsetY;
			int blockWidth, blockHeight;
			int trueX1, trueY1, trueX2, trueY2;
			m_ImgPartition.GetBlockInfoSimple(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			SetBlockState(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			this->PrintBlockState();

			clock_t tempTime =  clock();
			printf("Refine DataCost:\n");

			//PrintBlockState();
			//GetBundleDataCost(nearFrames, PList, pCurrentFrame, DataCost, labelImg);

			SDataCostParas pars;
			pars.blockHeight = blockHeight;
			pars.blockWidth = blockWidth;
			pars.dspLevels = dspLevel;
			pars.ImgHeight = ImgHeight;
			pars.ImgWidth  = ImgWidth;
			pars.m_dColorSigma = m_dColorSigma;
			pars.m_fColorMissPenalty = m_fColorMissPenalty;
			pars.m_fDataCostWeight = GetTrueDataCostWeight();
			pars.offsetX = offsetX;
			pars.offsetY = offsetY;
			pars.TrueX1 = m_iTrueX1;
			pars.TrueY1 = m_iTrueY1;
			pars.TrueX2 = m_iTrueX2;
			pars.TrueY2 = m_iTrueY2;

			pars.finalDspSigma = (m_dDspMax - m_dDspMin) * m_dDspSigma; //变换后再传入

			GpuGetBODataCost(count, ColorImgPitchBytes, DspImgPitchBytes, pars, DataCost.GetBits(), DataCost.GetWidth(), dataCostPitchedPtr, d_labelImg, d_labeImgPitch);
			GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);

			//pCurrentFrame->SaveDataCost(DataCost);
			//pCurrentFrame->ReadDataCost(DataCost, labelImg);
			//pCurrentFrame->SetDspImg(labelImg);
			//if(m_ifGenerateTempResult == true){
				//pCurrentFrame->SaveLabelImg(labelImg,  CVDRVideoFrame::TYPE::INIT, dspLevel);
				//system("pause");

				//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
				//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
				//system("pause");

				//GpuSetLabelImgByDataCost(pars, dataCostPitchedPtr, d_labelImg, d_labeImgPitch);
				//GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
				//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
				//system("pause");

				//GetBundleDataCost(nearFrames, PList, pCurrentFrame, DataCost, labelImg);
				//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
				//system("pause");
				//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
				//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
				//system("pause");
			//}
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

			printf("Refine By BP:\n");
			RefineDspByGpuBp(DataCost, dataCostPitchedPtr, labelImg, d_labelImg, NULL, d_labeImgPitch, true);
			GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);

			//RefineDspByGpuBp2(DataCost, dataCostPitchedPtr, labelImg, d_labelImg, NULL, d_labeImgPitch, true);

			if( m_iSubSample <= 1)
				pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
			//if(m_ifGenerateTempResult == true){
				//pCurrentFrame->SaveLabelImg(labelImg,  CVDRVideoFrame::TYPE::BO, dspLevel);
				//system("pause");

				//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
				//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::BO, dspLevel);
				//system("pause");
			//}
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

			if(m_iSubSample > 1){
				printf("Refine By Depth Expansion:\n");
				//PrintBlockState();
				//tmpLableImg = labelImg;

				GpuSetGpuLabelImgByGpuLabelImg(d_labelImg, d_tmpLableImg, d_labeImgPitch, offsetX, offsetY, offsetX+blockWidth-1, offsetY+blockHeight-1);
				_SuperResolutionRefineByGPU(count, 1, subDataCost, subDataCostPitchPtr, d_offsetImg, labelImg, d_tmpLableImg, d_labeImgPitch, ColorImgPitchBytes, DspImgPitchBytes);

				GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_tmpLableImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
				int newdspLevel = (dspLevel - 1) * m_iSubSample + 1;
				pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2, m_dDspMin, m_dDspMax, newdspLevel);

				//if(m_ifGenerateTempResult == true){
				//pCurrentFrame->SaveLabelImg(labelImg,  CVDRVideoFrame::TYPE::DeptExpansion, newdspLevel);
				//system("pause");
				//}
				printf("Time:%.3f s\n\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			}
		}

	GpuClearColorImg(count);
	GpuClearDspImg(count);
	GpuDeleMem(d_labelImg);
	GpuDeleMem(d_tmpLableImg);
	GpuDeleMem(d_offsetImg);

	if(m_ifGenerateTempResult == true){
		if(m_iSubSample > 1){
			int newdspLevel = (dspLevel - 1) * m_iSubSample + 1;
			pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);
		}
		else
			pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);
	}

	//pCurrentFrame->InitLabelImgByDspImg(labelImg, (dspLevel - 1) * m_iSubSample + 1);
	//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::BO, (dspLevel - 1) * m_iSubSample + 1);
	pCurrentFrame->SaveDspImg();
	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->Clear();
	pCurrentFrame->Clear();
}

void CDepthRecoverFuns::RefineDepthWithPfByGpu( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame,
	CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr, CDataCost& subDataCost, cudaPitchedPtr& subDataCostPitchPtr)
{
	int ImgHeight = CVDRVideoFrame::GetImgHeight();
	int ImgWidth = CVDRVideoFrame::GetImgWidth();

	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());
	//load
	pCurrentFrame->LoadColorImg();
	pCurrentFrame->LoadDepthImg();
	for(int i=0; i<nearFrames.size(); i++){
		nearFrames[i]->LoadDepthImg();
		nearFrames[i]->LoadColorImg();
	}

	int dspLevel = DataCost.GetChannel();

	ZIntImage labelImg;
	pCurrentFrame->InitLabelImgByDspImg(labelImg, dspLevel);
	//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::DeptExpansion, dspLevel);

	int count = nearFrames.size() + 1;
	//gpu
	size_t ColorImgPitchBytes;
	size_t DspImgPitchBytes;
	int * d_labelImg;
	int * d_tmpLableImg;
	int * d_offsetImg;
	size_t d_labeImgPitch;

	//color image
	unsigned char **colorImgs = new unsigned char *[count];
	colorImgs[0] = (unsigned char *)pCurrentFrame->m_pColorImg->GetMap();
	for(int i=1; i<count; i++)
		colorImgs[i] = (unsigned char *)nearFrames[i-1]->m_pColorImg->GetMap();
	int EffectWidthBytes = pCurrentFrame->m_pColorImg->GetEffectWidth();
	GpuSetColorImg(colorImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, &ColorImgPitchBytes);
	delete [] colorImgs;

	//Dsp image
	float **dspImgs = new float *[count];
	dspImgs[0] = (float *)pCurrentFrame->m_pDspImg->GetMap();
	for(int i=1; i<count; i++)
		dspImgs[i] = (float *)nearFrames[i-1]->m_pDspImg->GetMap();
	EffectWidthBytes = pCurrentFrame->m_pDspImg->GetEffectWidth();
	GpuSetDspImg(dspImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, &DspImgPitchBytes);
	//GpuGetDspImg(dspImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, DspImgPitchBytes);
	delete [] dspImgs;

	//camera parameters
	float* cameraParameters = new float[count * 21];
	pCurrentFrame->GetCameraParas(&(cameraParameters[0]));
	for(int i=1; i<count; i++)
		nearFrames[i-1]->GetCameraParas(&(cameraParameters[i * 21]));
	GpuSetCameraParas(cameraParameters, count);
	delete[] cameraParameters;

	//dspV
	GpuSetDspV(&m_fDspV[0], dspLevel);

	//labelImg
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_tmpLableImg), &d_labeImgPitch);
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_labelImg), &d_labeImgPitch);
	GpuSetGpuLabelImgByLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, 0, 0, ImgWidth-1, ImgHeight - 1);

	//offsetImg
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_offsetImg), &d_labeImgPitch);

	//debug test
	//std::vector<Wml::Matrix4d> PList(nearFrames.size());
	//GetPList(PList, pCurrentFrame, nearFrames);

	int blockCountX = m_ImgPartition.getBlockCountX();
	int blockCountY = m_ImgPartition.getBlockCountY();
	for(int blockIdy = 0; blockIdy < blockCountY; blockIdy++)
		for(int blockIdX = 0; blockIdX < blockCountX; blockIdX++){
			int offsetX, offsetY;
			int blockWidth, blockHeight;
			int trueX1, trueY1, trueX2, trueY2;
			m_ImgPartition.GetBlockInfoSimple(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			SetBlockState(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			this->PrintBlockState();

			clock_t tempTime =  clock();
			printf("Refine DataCost With PlainFitting:\n");

			//PrintBlockState();
			//GetBundleDataCost(nearFrames, PList, pCurrentFrame, DataCost, labelImg);

			SDataCostParas pars;
			pars.blockHeight = blockHeight;
			pars.blockWidth = blockWidth;
			pars.dspLevels = dspLevel;
			pars.ImgHeight = ImgHeight;
			pars.ImgWidth  = ImgWidth;
			pars.m_dColorSigma = m_dColorSigma;
			pars.m_fColorMissPenalty = m_fColorMissPenalty;
			pars.m_fDataCostWeight = GetTrueDataCostWeight();
			pars.offsetX = offsetX;
			pars.offsetY = offsetY;
			pars.TrueX1 = m_iTrueX1;
			pars.TrueY1 = m_iTrueY1;
			pars.TrueX2 = m_iTrueX2;
			pars.TrueY2 = m_iTrueY2;

			pars.finalDspSigma = (m_dDspMax - m_dDspMin) * m_dDspSigma; //变换后再传入

			GpuGetBODataCost(count, ColorImgPitchBytes, DspImgPitchBytes, pars, DataCost.GetBits(), DataCost.GetWidth(), dataCostPitchedPtr, d_labelImg, d_labeImgPitch);
			GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);

			//pCurrentFrame->SaveDataCost(DataCost);
			//pCurrentFrame->ReadDataCost(DataCost, labelImg);
			//pCurrentFrame->SetDspImg(labelImg);
			//if(m_ifGenerateTempResult == true){
			//pCurrentFrame->SaveLabelImg(labelImg,  CVDRVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");

			//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
			//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");

			//GpuSetLabelImgByDataCost(pars, dataCostPitchedPtr, d_labelImg, d_labeImgPitch);
			//GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
			//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");

			//GetBundleDataCost(nearFrames, PList, pCurrentFrame, DataCost, labelImg);
			//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");
			//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
			//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::INIT, dspLevel);
			//system("pause");
			//}
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

			printf("Refine By BP:\n");
			RefineDspByGpuBp(DataCost, dataCostPitchedPtr, labelImg, d_labelImg, NULL, d_labeImgPitch, true);
			GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);

			pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
			printf("Refine By Segmentation:\n");
			CDepthRecoverFuns::GetInstance()->RefineDspBySegm(pCurrentFrame, labelImg, DataCost);
			GpuSetGpuLabelImgByLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);

			//if(m_ifGenerateTempResult == true)
			//	pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::DeptExpansion, dspLevel);
			printf("Time:%.3f s\n\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
	//		pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
		//	if( m_iSubSample <= 1)
			//if(m_ifGenerateTempResult == true){
			//pCurrentFrame->SaveLabelImg(labelImg,  CVDRVideoFrame::TYPE::BO, dspLevel);
			//system("pause");
			//DataCost.SetLabelImg(offsetX, offsetY, blockWidth, blockHeight, labelImg);
			//pCurrentFrame->SaveLabelImg(labelImg,  CVideoFrame::TYPE::BO, dspLevel);
			//system("pause");
			//}
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			tempTime = clock();

			if(m_iSubSample > 1){
			//	SaveZimg(labelImg,"before");
				printf("Refine By Depth Expansion:\n");
				//PrintBlockState();
				//tmpLableImg = labelImg;

				GpuSetGpuLabelImgByGpuLabelImg(d_labelImg, d_tmpLableImg, d_labeImgPitch, offsetX, offsetY, offsetX+blockWidth-1, offsetY+blockHeight-1);
				_SuperResolutionRefineByGPU(count, 1, subDataCost, subDataCostPitchPtr, d_offsetImg, labelImg, d_tmpLableImg, d_labeImgPitch, ColorImgPitchBytes, DspImgPitchBytes);

				GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_tmpLableImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
				int newdspLevel = (dspLevel - 1) * m_iSubSample + 1;
				pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2, m_dDspMin, m_dDspMax, newdspLevel);

				printf("Time:%.3f s\n\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
			}
		//	SaveZimg(labelImg,"after");
		}

		GpuClearColorImg(count);
		GpuClearDspImg(count);
		GpuDeleMem(d_labelImg);
		GpuDeleMem(d_tmpLableImg);
		GpuDeleMem(d_offsetImg);

		if(m_ifGenerateTempResult == true){
			if(m_iSubSample > 1){
				int newdspLevel = (dspLevel - 1) * m_iSubSample + 1;
				pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);
			}
			else
			{
				//pCurrentFrame->InitLabelImgByDspImg(labelImg,dspLevel);
				pCurrentFrame->SaveDspLabelImg(CVDRVideoFrame::BO,m_dDspMin,m_dDspMax);
			}
		}

		//pCurrentFrame->InitLabelImgByDspImg(labelImg, (dspLevel - 1) * m_iSubSample + 1);
		//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::BO, (dspLevel - 1) * m_iSubSample + 1);
		pCurrentFrame->SaveDspImg();
		for(int i=0; i<nearFrames.size(); i++)
			nearFrames[i]->Clear();
		pCurrentFrame->Clear();
}

void CDepthRecoverFuns::UpSampling( std::vector<CVDRVideoFrame*>& FwFrames, std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame )
{
	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());

	//must load color image first!! then load Depth Image!!
	pCurrentFrame->LoadColorImg();
	pCurrentFrame->LoadDepthImg();
	for(int i=0; i<nearFrames.size(); i++){
		nearFrames[i]->LoadColorImg();
		//nearFrames[i]->LoadDepthImg();
	}

	std::vector<Wml::Matrix4d> PList(nearFrames.size());
	GetPList(PList, pCurrentFrame, nearFrames);

	//debug
	ZIntImage labelImg;
	pCurrentFrame->InitLabelImgByDspImg(labelImg, m_iDspLevel);

	//EdgeDepthInit(pCurrentFrame, nearFrames, PList, labelImg);
	//EdgeDepthInitGlobal(pCurrentFrame, nearFrames, PList, labelImg);
	EdgeGCRefine(pCurrentFrame, nearFrames, PList, labelImg, false);

	//pCurrentFrame->InitLabelImgByDspImg(labelImg, m_iDspLevel);
	pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::DeptExpansion, m_iDspLevel);
	pCurrentFrame->SaveDspImg();

	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->Clear();
	pCurrentFrame->Clear();
}

void CDepthRecoverFuns::UpSampling( CVDRVideoFrame* pCurrentFrame,float scale )
{
	pCurrentFrame->UpSampling(scale);
	//debug
	//ZIntImage labelImg;
	//pCurrentFrame->InitLabelImgByDspImg(labelImg, m_iDspLevel);
	//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::DeptExpansion, m_iDspLevel);
	pCurrentFrame->SaveDspImg();
	pCurrentFrame->Clear();
}

void CDepthRecoverFuns::RefineEdgeDepth( std::vector<CVDRVideoFrame*>& FwFrames, std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame )
{
	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());

	//must load color image first!! then load Depth Image!!
	pCurrentFrame->LoadColorImg();
	pCurrentFrame->LoadDepthImg();
	for(int i=0; i<nearFrames.size(); i++){
		nearFrames[i]->LoadColorImg();
		nearFrames[i]->LoadDepthImg();
	}

	std::vector<Wml::Matrix4d> PList(nearFrames.size());
	GetPList(PList, pCurrentFrame, nearFrames);

	ZIntImage labelImg;
	pCurrentFrame->InitLabelImgByDspImg(labelImg, m_iDspLevel);

	//EdgeDepthRefine(pCurrentFrame, nearFrames, PList, labelImg);
	EdgeGCRefine(pCurrentFrame, nearFrames, PList, labelImg, true);

	pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::DeptExpansion, m_iDspLevel);
	pCurrentFrame->SaveDspImg();

	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->Clear();
	pCurrentFrame->Clear();
}

void CDepthRecoverFuns::SuperResolutionRefine( std::vector<CVDRVideoFrame*>& FwFrames, std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame)
{
	if(m_iSubSample <= 1){
		printf("no nead to do SuperResolutionRefine\n");
		return;
	}

	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());

	//must load color image first!! then load Depth Image!!
	pCurrentFrame->LoadColorImg();
	pCurrentFrame->LoadDepthImg();
	for(int i=0; i<nearFrames.size(); i++){
		nearFrames[i]->LoadColorImg();
		nearFrames[i]->LoadDepthImg();
	}

	std::vector<Wml::Matrix4d> PList(nearFrames.size());
	GetPList(PList, pCurrentFrame, nearFrames);

	ZIntImage offsetImg, labelImg;
	pCurrentFrame->InitLabelImgByDspImg(labelImg, m_iDspLevel);
	offsetImg.CreateAndInit(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), 1, -1);

	//debug
	//EdgeDepthInit(pCurrentFrame, nearFrames, PList, labelImg);
	//pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::DeptExpansion, m_iDspLevel);
	//exit(0);

	int width =  CVDRVideoFrame::GetImgWidth();
	int height = CVDRVideoFrame::GetImgHeight();
	ZIntImage tmpLableImg;

	//pCurrentFrame->getLableImgFrmDspImg(labelImg, m_iDspLevel, m_dDspMin, m_dDspMax);
	//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::DeptExpansion, m_iDspLevel);
	//exit(0);

	int blockCountX = m_ImgPartition.getBlockCountX();
	int blockCountY = m_ImgPartition.getBlockCountY();

	int LayerCount = m_iDspLevel - 1;
	for(int i=0; i<m_iSubIter; i++)
		LayerCount *= m_iSubSample;

	for(int blockIdy = 0; blockIdy < blockCountY; blockIdy++)
		for(int blockIdX = 0; blockIdX < blockCountX; blockIdX++){
			int offsetX, offsetY;
			int blockWidth, blockHeight;
			int trueX1, trueY1, trueX2, trueY2;
			m_ImgPartition.GetBlockInfoSimple(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			SetBlockState(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);

			tmpLableImg = labelImg;
			PrintBlockState();
			_SuperResolutionRefine(pCurrentFrame, nearFrames, m_iSubIter, PList, offsetImg, tmpLableImg);
			pCurrentFrame->SetDspImg(tmpLableImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2, m_dDspMin, m_dDspMax, LayerCount + 1);

			//printf("Save offsetImg!!!\n");
			pCurrentFrame->SaveLabelImg(tmpLableImg, CVDRVideoFrame::TYPE::DeptExpansion, LayerCount + 1);
			//pCurrentFrame->SaveLabelImg(offsetImg, CVDRVideoFrame::TYPE::DeptExpansion, LayerCount + 1);
		}

	pCurrentFrame->InitLabelImgByDspImg(labelImg, LayerCount + 1);
	pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::DeptExpansion, LayerCount + 1);

	pCurrentFrame->SaveDspImg();
	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->Clear();
	pCurrentFrame->Clear();
}
void CDepthRecoverFuns::SuperResolutionRefineByGpu( std::vector<CVDRVideoFrame*>& FwFrames, std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& subDataCost, cudaPitchedPtr& subDataCostPitchPtr){
	int ImgHeight = CVDRVideoFrame::GetImgHeight();
	int ImgWidth = CVDRVideoFrame::GetImgWidth();

	std::vector<CVDRVideoFrame*> nearFrames(FwFrames.begin(), FwFrames.end());
	nearFrames.insert(nearFrames.end(), BwFrames.begin(), BwFrames.end());
	//load
	pCurrentFrame->LoadColorImg();
	pCurrentFrame->LoadDepthImg();

	for(int i=0; i<nearFrames.size(); i++){
		nearFrames[i]->LoadColorImg();
		nearFrames[i]->LoadDepthImg();
	}

	ZIntImage labelImg;
	pCurrentFrame->InitLabelImgByDspImg(labelImg, m_iDspLevel);
	//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::DeptExpansion, m_iDspLevel);

	int count = nearFrames.size() + 1;
	//gpu
	size_t ColorImgPitchBytes;
	size_t DspImgPitchBytes;
	int * d_labelImg;
	int * d_tmpLableImg;
	int * d_offsetImg;
	size_t d_labeImgPitch;

	//color image
	unsigned char **colorImgs = new unsigned char *[count];
	colorImgs[0] = (unsigned char *)pCurrentFrame->m_pColorImg->GetMap();
	for(int i=1; i<count; i++)
		colorImgs[i] = (unsigned char *)nearFrames[i-1]->m_pColorImg->GetMap();
	int EffectWidthBytes = pCurrentFrame->m_pColorImg->GetEffectWidth();
	GpuSetColorImg(colorImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, &ColorImgPitchBytes);
	delete [] colorImgs;

	//Dsp image
	float **dspImgs = new float *[count];
	dspImgs[0] = (float *)pCurrentFrame->m_pDspImg->GetMap();
	for(int i=1; i<count; i++)
		dspImgs[i] = (float *)nearFrames[i-1]->m_pDspImg->GetMap();
	EffectWidthBytes = pCurrentFrame->m_pDspImg->GetEffectWidth();
	GpuSetDspImg(dspImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, &DspImgPitchBytes);
	//GpuGetDspImg(dspImgs, EffectWidthBytes, ImgWidth, ImgHeight, count, DspImgPitchBytes);
	delete [] dspImgs;

	//camera parameters
	float* cameraParameters = new float[count * 21];
	pCurrentFrame->GetCameraParas(&(cameraParameters[0]));
	for(int i=1; i<count; i++)
		nearFrames[i-1]->GetCameraParas(&(cameraParameters[i * 21]));
	GpuSetCameraParas(cameraParameters, count);
	delete[] cameraParameters;

	////dspV
	//GpuSetDspV(&m_fDspV[0], m_iDspLevel);

	//labelImg
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_tmpLableImg), &d_labeImgPitch);
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_labelImg), &d_labeImgPitch);
	GpuSetGpuLabelImgByLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_labelImg, d_labeImgPitch, 0, 0, ImgWidth-1, ImgHeight - 1);

	//offsetImg
	GpuNewLabelImg(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), (void **)(&d_offsetImg), &d_labeImgPitch);

	//test
	//std::vector<Wml::Matrix4d> PList(nearFrames.size());
	//GetPList(PList, pCurrentFrame, nearFrames);

	int blockCountX = m_ImgPartition.getBlockCountX();
	int blockCountY = m_ImgPartition.getBlockCountY();

	int LayerCount = m_iDspLevel - 1;
	for(int i=0; i<m_iSubIter; i++)
		LayerCount *= m_iSubSample;

	for(int blockIdy = 0; blockIdy < blockCountY; blockIdy++)
		for(int blockIdX = 0; blockIdX < blockCountX; blockIdX++){
			int offsetX, offsetY;
			int blockWidth, blockHeight;
			int trueX1, trueY1, trueX2, trueY2;
			m_ImgPartition.GetBlockInfoSimple(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			SetBlockState(trueX1, trueY1, trueX2, trueY2, offsetX, offsetY, blockWidth, blockHeight, blockIdX, blockIdy);
			PrintBlockState();

			clock_t tempTime =  clock();

			//PrintBlockState();
			//GetBundleDataCost(nearFrames, PList, pCurrentFrame, DataCost, labelImg);

			printf("Refine By Depth Expansion:\n");
			//tmpLableImg = labelImg;

			GpuSetGpuLabelImgByGpuLabelImg(d_labelImg, d_tmpLableImg, d_labeImgPitch, offsetX, offsetY, offsetX+blockWidth-1, offsetY+blockHeight-1);
			_SuperResolutionRefineByGPU(count, m_iSubIter,subDataCost, subDataCostPitchPtr, d_offsetImg, labelImg, d_tmpLableImg, d_labeImgPitch, ColorImgPitchBytes, DspImgPitchBytes);

			GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), d_tmpLableImg, d_labeImgPitch, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
			pCurrentFrame->SetDspImg(labelImg, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2, m_dDspMin, m_dDspMax, LayerCount + 1);

			//if(m_ifGenerateTempResult == true){
				//pCurrentFrame->SaveDspExpanLabelImg(tmpLableImg, m_iDspLevel, m_iOffsetX, m_iOffsetY, m_iBlockWidth, m_iBlockHeight, (m_iDspLevel - 1) * m_iSubSample + 1);
			//}
			//if(m_ifGenerateTempResult == true){
			//	pCurrentFrame->SaveLabelImg(tmpLableImg,  CVideoFrame::TYPE::INIT, (m_iDspLevel - 1) * m_iSubSample + 1);
			//	system("pause");
			//}
			printf("Time:%.3f s\n",(double)(clock()-tempTime)/CLOCKS_PER_SEC);
		}

	GpuClearColorImg(count);
	GpuClearDspImg(count);
	GpuDeleMem(d_labelImg);
	GpuDeleMem(d_tmpLableImg);
	GpuDeleMem(d_offsetImg);

	if(m_ifGenerateTempResult == true){
		pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::DeptExpansion, LayerCount + 1);
		//pCurrentFrame->SaveLabelImg(labelImg, CVideoFrame::TYPE::DeptExpansion, m_iDspLevel);
	}

	pCurrentFrame->SaveDspImg();
	for(int i=0; i<nearFrames.size(); i++)
		nearFrames[i]->Clear();
	pCurrentFrame->Clear();
}

void CDepthRecoverFuns::_SuperResolutionRefine(CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& nearFrames,int iterCount,
											   std::vector<Wml::Matrix4d>& PList, ZIntImage& offsetImg, ZIntImage& labelImg)
{
	int subDeplevel = 2 * m_iSubSample + 1;
	//float * subDataCost = new float[m_iBlockWidth * m_iBlockHeight * subDeplevel];

	CDataCost subDataCost(m_iBlockWidth, m_iBlockHeight, subDeplevel);
	//ZIntImage offsetImg;
	//offsetImg.CreateAndInit(m_iBlockWidth, m_iBlockHeight);

	//====================protect=======================
	//double backUpprjSigma = m_dPrjSigma;
	double backUpdspSigma = m_dDspSigma;

	float backUpDataCostWeight = m_fDataCostWeight;
	float backUpDiscK = m_fDiscK;
	int LayerCount = m_iDspLevel - 1;

	m_fDataCostWeight *= (float)m_iSubSample /(m_iDspLevel - 1); //(subDeplevel - 1.0F) / (m_iDspLevel - 1);
	m_fDiscK *= m_iSubSample; //(subDeplevel - 1.0F) / (m_iDspLevel - 1);

	CParallelManager pm(m_iCpuThreads);
	for(int iPass = 0; iPass < iterCount; iPass++){
		printf("iter: %d/%d\n", iPass+1, iterCount);
		LayerCount *= m_iSubSample;

		//m_dDspSigma = max(m_dMinDspSigma, m_dDspSigma / m_iSubSample);
		////m_dPrjSigma = max(m_dMinPrjSigma, 2 * m_dPrjSigma / m_iSubSample);

		printf("Layer count:%d, subDspLevel:%d, dspSigma:%f DataCost weight:%f, DspMin:%lf, DspMax:%lf, colorSigma:%lf, dspSigma:%lf\n",
			LayerCount, subDeplevel, m_dDspSigma, GetTrueDataCostWeight(), m_dDspMin, m_dDspMax, m_dColorSigma, m_dDspSigma);

		for(int j=0; j<m_iBlockHeight; ++j){
			CSubRefineWorkUnit* pWorkUnit
				= new CSubRefineWorkUnit(pCurrentFrame, nearFrames, PList,
				labelImg, subDataCost, j, LayerCount+1, m_iSubSample, m_dDspMin, m_dDspMax, offsetImg, m_iBlockWidth);

			pm.EnQueue(pWorkUnit);
		}
		pm.Run();

		//ZIntImage labelImgTmp;
		//labelImgTmp.Create(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(), 1);
		//for(int j=0; j<m_iBlockHeight; ++j)
		//	for(int i=0; i<m_iBlockWidth; ++i){
		//		labelImgTmp.at(m_iOffsetX + i, m_iOffsetY + j) = labelImg.at(m_iOffsetX + i, m_iOffsetY + j) + offsetImg.at(m_iOffsetX + i, m_iOffsetY + j);
		//	}
		//pCurrentFrame->SaveLabelImg(labelImgTmp, CVDRVideoFrame::TYPE::DeptExpansion, LayerCount + 1);
		//system("pause");
		printf("\n");
//========================================================================================
		//ZIntImage labelImgTest(labelImg);

		//ZCubeFloatImage subDataCostTest;
		//subDataCostTest.CreateAndInit(m_iBlockWidth, m_iBlockHeight, subDeplevel);
		//for(int y = 0; y< m_iBlockHeight; y++)
		//	for(int x = 0; x < m_iBlockWidth; x++)
		//		for(int k = 0; k< subDeplevel; k++)
		//			subDataCostTest.SetPixel(x, y, k, subDataCost.GetValueAt(x, y, k));
		//NumericalAlgorithm::CExWRegularGridBP stereoBP;
		//ZFloatImage disMap, truncMap;
		//disMap.CreateAndInit(m_iBlockWidth, m_iBlockHeight, 4, 1);
		//truncMap.CreateAndInit(m_iBlockWidth, m_iBlockHeight,4, GetTrueDiscK());
		//stereoBP.m_iThreadCount = 1;
		//stereoBP.m_iMaxIter = 4;
		//stereoBP.Solve(subDataCost,  &offsetImg, labelImgTest, subDataCostTest, labelImg, offsetImg, disMap, truncMap);
//========================================================================================

		//RefineDspByBP(subDataCost, labelImg, false, NULL);
		RefineDspByBP(subDataCost, labelImg, false, &offsetImg);

		//Careful!!! Should offset!!!
		for(int j=0; j<m_iBlockHeight; ++j)
			for(int i=0; i<m_iBlockWidth; ++i){
				labelImg.at(m_iOffsetX + i, m_iOffsetY + j) += offsetImg.at(m_iOffsetX + i, m_iOffsetY + j);
			}
		//pCurrentFrame->SaveLabelImg(labelImg, CVDRVideoFrame::TYPE::DeptExpansion, LayerCount + 1);
		//system("pause");

		m_dDspSigma = max(m_dMinDspSigma, 2 * m_dDspSigma / m_iSubSample);
	}

	//================recover==========================
	//m_dPrjSigma = backUpprjSigma;
	m_dDspSigma = backUpdspSigma;
	m_fDataCostWeight = backUpDataCostWeight;
	m_fDiscK = backUpDiscK;
}
void CDepthRecoverFuns::_SuperResolutionRefineByGPU(int frameCount, int iterCount, CDataCost& subDataCost, cudaPitchedPtr& subDataCostPitchPtr, void* offsetImg, ZIntImage& labelImg,
													void* tmpLableImg, size_t labeImgPitchBytes, size_t ColorImgPitchBytes, size_t DspImgPitchBytes){
	int subDeplevel = 2 * m_iSubSample + 1;

	//====================protect=======================
	//double backUpprjSigma = m_dPrjSigma;
	double backUpdspSigma = m_dDspSigma;
	float backUpDiscK = m_fDiscK;
	int LayerCount = m_iDspLevel - 1;
	SDataCostParas pars;
	pars.blockHeight = m_iBlockHeight;
	pars.blockWidth = m_iBlockWidth;
	pars.dspLevels = subDataCost.GetChannel();
	pars.finalDspSigma = (m_dDspMax - m_dDspMin) * m_dDspSigma; //变换后再传入
	pars.ImgHeight = labelImg.GetHeight();
	pars.ImgWidth  = labelImg.GetWidth();
	pars.m_dColorSigma = m_dColorSigma;
	pars.m_fColorMissPenalty = m_fColorMissPenalty;
	pars.offsetX = m_iOffsetX;
	pars.offsetY = m_iOffsetY;
	pars.TrueX1 = m_iTrueX1;
	pars.TrueY1 = m_iTrueY1;
	pars.TrueX2 = m_iTrueX2;
	pars.TrueY2 = m_iTrueY2;
	pars.m_fDataCostWeight = GetTrueDataCostWeight() * m_iSubSample / (m_iDspLevel - 1);

	m_fDiscK *= m_iSubSample ; //(subDeplevel - 1.0F) / (m_iDspLevel - 1);

	for(int iPass = 0; iPass < iterCount; iPass++){
		printf("Iter: %d/%d\n", iPass+1, iterCount);
		LayerCount *= m_iSubSample;

		//m_dDspSigma = max(m_dMinDspSigma, m_dDspSigma / m_iSubSample);
		//pars.finalDspSigma = (m_dDspMax - m_dDspMin) * m_dDspSigma; //变换后再传入

		printf("Layer count:%d, subDspLevel:%d, dspSigma:%f DataCost weight:%f, DspMin:%lf, DspMax:%lf, colorSigma:%lf, dspSigma:%lf\n",
			LayerCount, subDeplevel, m_dDspSigma, pars.m_fDataCostWeight, m_dDspMin, m_dDspMax, m_dColorSigma, m_dDspSigma);

			//CSubRefineWorkUnit* pWorkUnit
			//	= new CSubRefineWorkUnit(pCurrentFrame, nearFrames, PList,
			//	labelImg, subDataCost, j, LayerCount+1, m_iSubSample, m_dDspMin, m_dDspMax, offsetImg, m_iBlockWidth);

		GpuSubRefine(frameCount, ColorImgPitchBytes, DspImgPitchBytes, pars, subDataCostPitchPtr, tmpLableImg, labeImgPitchBytes, LayerCount+1, m_iSubSample, m_dDspMin, m_dDspMax, offsetImg);

		//ZIntImage labelImgTmp;
		//labelImgTmp.Create(width, height, 1);
		//for(int j=0; j<height; ++j)
		//	for(int i=0; i<width; ++i){
		//		labelImgTmp.at(i,j) = labelImg.at(i, j) + offsetImg.at(i,j);
		//	}
		//pCurrentFrame->SaveLabelImg(labelImgTmp, CVideoFrame::TYPE::INITBYCOLOR, LayerCount + 1);

		RefineDspByGpuBp(subDataCost, subDataCostPitchPtr, labelImg, tmpLableImg, offsetImg, labeImgPitchBytes, false);
		//Careful!!! Should offset!!!
		//for(int j=0; j<m_iBlockHeight; ++j)
		//	for(int i=0; i<m_iBlockWidth; ++i){
		//		labelImg.at(m_iOffsetX + i, m_iOffsetY + j) += offsetImg.at(i,j);
		//	}

		m_dDspSigma = max(m_dMinDspSigma, 2 * m_dDspSigma / m_iSubSample);
		pars.finalDspSigma = (m_dDspMax - m_dDspMin) * m_dDspSigma; //变换后再传入
	}
	GpuSetLabelImgByGpuLabelImg(labelImg.GetMap(), labelImg.GetEffectWidth(), tmpLableImg, labeImgPitchBytes, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);

	//================recover==========================
	//m_dPrjSigma = backUpprjSigma;
	m_dDspSigma = backUpdspSigma;
	m_fDiscK = backUpDiscK;
}

void CDepthRecoverFuns::EdgeDepthInit( CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& NearFrames, std::vector<Wml::Matrix4d>& PList, ZIntImage& labelImg)
{
	int radius = 1;
	int ImgWidth = CVDRVideoFrame::GetImgWidth();
	int ImgHeight = CVDRVideoFrame::GetImgHeight();

	std::set<SPoint> EdgePoints;
	pCurrentFrame->GetEdgePoints(EdgePoints, 0);

	std::set<int> labels;
	vector<float> dspV;
	int u, v, px, py;

	CDataCostUnit dataCosti;
	CDataCost dataCost, uMsg, dMsg, lMsg, rMsg;
	for(std::set<SPoint>::iterator it = EdgePoints.begin(); it != EdgePoints.end(); it++){
		//labelImg.at(it->x, it->y, 0) = 0;
		//continue;
		u = it->x;
		v = it->y;

		int left = u-radius >= 0 ? u-radius : 0;
		int right = u+radius <ImgWidth ? u+radius : ImgWidth - 1;
		int up = v-radius>=0? v-radius: 0;
		int down = v+radius<ImgHeight? v+radius : ImgHeight-1;

		labels.clear();
		for(px = left; px<right; px++){
			for(py = up; py<down; py++){
				int value = labelImg.at(px, py);
				labels.insert(value);
				//if(value-1>=0)
				//	labels.insert(value-1);
				//if(value+1<m_iDspLevel)
				//	labels.insert(value+1);
			}
		}
		int DspLevelCount = labels.size();

		dspV.resize(DspLevelCount);
		int loc = 0;
		for(set<int>::iterator it = labels.begin();  it!=labels.end(); it++)
			dspV[loc++] = GetDspFromDspLeveli(*it);

		int bestLabel;
		dataCost.Create(right - left + 1, down - up + 1, DspLevelCount);

		std::set<SPoint>::iterator ti;
		SPoint pi;

		for(px = left; px<right; px++){
			for(py = up; py<down; py++){
				dataCost.GetDataCostUnit(px-left, py-up, dataCosti);
				GetInitialDataCostAt(pCurrentFrame, px, py, dataCosti, NearFrames, bestLabel, dspV, PList);

				//if(px>left && px<right && py>up && py<down){
				//	labelImg.SetPixel(px, py, 0, GetLevelFromDspi(dspV[bestLabel], m_iDspLevel));
				//	pCurrentFrame->SetDspAt(px, py, dspV[bestLabel]);
				//}
				labelImg.SetPixel(px, py, 0, GetLevelFromDspi(dspV[bestLabel], m_iDspLevel));
				pCurrentFrame->SetDspAt(px, py, dspV[bestLabel]);
			}
		}
		/*if(right-left+1<=2 || down-up+1 <= 2)
			continue;

		uMsg.Create(right - left + 1, down - up + 1, DspLevelCount, true);
		dMsg.Create(right - left + 1, down - up + 1, DspLevelCount, true);
		lMsg.Create(right - left + 1, down - up + 1, DspLevelCount, true);
		rMsg.Create(right - left + 1, down - up + 1, DspLevelCount, true);

		EdgeSmooth(dataCost, uMsg, dMsg, lMsg, rMsg, labelImg, left, right, up, down, DspLevelCount, 4, labels);
		for(px = left+1; px<=right-1; px++){
			for(py = up+1; py<=down-1; py++){
				int labeli = labelImg.at(px, py);
				float dsp = dspV[labeli];
				labelImg.SetPixel(px, py, 0, GetLevelFromDspi(dsp, m_iDspLevel));
				pCurrentFrame->SetDspAt(px, py, dsp);
				pi.x = px;
				pi.y = py;
				pi.mark = false;
				ti = EdgePoints.find(pi);
				if(ti!=EdgePoints.end() && ti != it)
					EdgePoints.erase(ti);
			}
		}*/
	}
}

void CDepthRecoverFuns::EdgeDepthGCInit( CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& NearFrames, std::vector<Wml::Matrix4d>& PList, ZIntImage& labelImg){
	int ImgWidth = CVDRVideoFrame::GetImgWidth();
	int ImgHeight = CVDRVideoFrame::GetImgHeight();

	std::set<SPoint> EdgePointsSet;
	pCurrentFrame->GetEdgePoints(EdgePointsSet, 1);
	std::vector<SPoint>EdgePoints(EdgePointsSet.begin(), EdgePointsSet.end());
	int PointsCount = EdgePoints.size();

	std::set<int> labels;
	int u, v;
	for(std::vector<SPoint>::iterator it = EdgePoints.begin(); it!=EdgePoints.end(); it++){
		labels.insert(labelImg.at(it->x, it->y));
	}
	int DspLevelCount = labels.size();

	vector<float> dspV;
	dspV.resize(DspLevelCount);
	int loc = 0;
	for(set<int>::iterator it = labels.begin();  it!=labels.end(); it++)
		dspV[loc++] = GetDspFromDspLeveli(*it);

	float* dataCost = new float[EdgePoints.size() * DspLevelCount];
	CDataCostUnit dataCosti;

	for(int pointIndex = 0; pointIndex<PointsCount; pointIndex++){
		//labelImg.at(EdgePoints[pointIndex].x, EdgePoints[pointIndex].y, 0) = 0;
		//continue;

		u = EdgePoints[pointIndex].x;
		v = EdgePoints[pointIndex].y;

		dataCosti.Init(dataCost + pointIndex * DspLevelCount, 1, DspLevelCount);

		int bestLabel;
		GetInitialDataCostAt(pCurrentFrame, u, v, dataCosti, NearFrames, bestLabel, dspV, PList);
		labelImg.SetPixel(u, v, 0, GetLevelFromDspi(dspV[bestLabel], m_iDspLevel));
		pCurrentFrame->SetDspAt(u, v, dspV[bestLabel]);
	}

	/*EdgeGCSmooth(dataCost, labelImg, labels, EdgePoints);

	for(int pointIndex =0; pointIndex<PointsCount; pointIndex++){
		u = EdgePoints[pointIndex].x;
		v = EdgePoints[pointIndex].y;

		int labeli = labelImg.at(u, v);
		float dsp = dspV[labeli];
		labelImg.SetPixel(u, v, 0, GetLevelFromDspi(dsp, m_iDspLevel));
		pCurrentFrame->SetDspAt(u, v, dsp);
	}*/

	delete [] dataCost;
}

void CDepthRecoverFuns::EdgeDepthRefine( CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& NearFrames, std::vector<Wml::Matrix4d>& PList, ZIntImage& labelImg){
	int radius = 2;
	int ImgWidth = CVDRVideoFrame::GetImgWidth();
	int ImgHeight = CVDRVideoFrame::GetImgHeight();

	std::set<SPoint> EdgePoints;
	pCurrentFrame->GetEdgePoints(EdgePoints, 1);

	std::set<int> labels;
	int u, v, px, py;
	vector<float> dspV;

	CDataCost dataCost, uMsg, dMsg, lMsg, rMsg;
	CDataCostUnit dataCosti;

	for(std::set<SPoint>::iterator it = EdgePoints.begin(); it != EdgePoints.end(); it++){
		//labelImg.at(it->x, it->y, 0) = 0;
		//continue;
		u = it->x;
		v = it->y;

		int left = u-radius >= 0 ? u-radius : 0;
		int right = u+radius <ImgWidth ? u+radius : ImgWidth - 1;
		int up = v-radius>=0? v-radius: 0;
		int down = v+radius<ImgHeight? v+radius : ImgHeight-1;

		labels.clear();
		for(px = left; px<=right; px++){
			for(py = up; py<=down; py++){
				int value = labelImg.at(px, py);
				labels.insert(value);
				//if(value-1>=0)
				//	labels.insert(value-1);
				//if(value+1<m_iDspLevel)
				//	labels.insert(value+1);
			}
		}
		int DspLevelCount = labels.size();
		dspV.resize(DspLevelCount);
		int loc = 0;
		for(set<int>::iterator it = labels.begin();  it!=labels.end(); it++)
			dspV[loc++] = GetDspFromDspLeveli(*it);

		dataCost.Create(right - left + 1, down - up + 1, DspLevelCount);
		uMsg.Create(right - left + 1, down - up + 1, DspLevelCount, true);
		dMsg.Create(right - left + 1, down - up + 1, DspLevelCount, true);
		lMsg.Create(right - left + 1, down - up + 1, DspLevelCount, true);
		rMsg.Create(right - left + 1, down - up + 1, DspLevelCount, true);

		int bestLabel;

		for(px = left; px<=right; px++){
			for(py = up; py<=down; py++){
				dataCost.GetDataCostUnit(px-left, py-up, dataCosti);
				GetBundleDataCostAt(pCurrentFrame, px, py, dataCosti, NearFrames, bestLabel, dspV, PList);
				if(px>left && px<right && py>up && py<down){
					labelImg.SetPixel(px, py, 0, GetLevelFromDspi(dspV[bestLabel], m_iDspLevel));
					pCurrentFrame->SetDspAt(px, py, dspV[bestLabel]);
				}
			}
		}
		if(right-left+1<=2 || down-up+1 <= 2)
			continue;

		EdgeSmooth(dataCost, uMsg, dMsg, lMsg, rMsg, labelImg, left, right, up, down, DspLevelCount, 4, labels);
		std::set<SPoint>::iterator ti;
		SPoint pi;
		for(px = left+1; px<=right-1; px++){
			for(py = up+1; py<=down-1; py++){
				int labeli = labelImg.at(px, py);
				float dsp = dspV[labeli];
				labelImg.SetPixel(px, py, 0, GetLevelFromDspi(dsp, m_iDspLevel));
				pCurrentFrame->SetDspAt(px, py, dsp);
				pi.x = px;
				pi.y = py;
				ti = EdgePoints.find(pi);
				if(ti!=EdgePoints.end() && ti != it)
					EdgePoints.erase(ti);
			}
		}
	}
}

void CDepthRecoverFuns::EdgeGCRefine(CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& NearFrames, std::vector<Wml::Matrix4d>& PList, ZIntImage& labelImg, bool bo){
	int boxWidth = 15;
	int kenalRadius = (boxWidth-1)/2 - 1;

	int ImgWidth = CVDRVideoFrame::GetImgWidth();
	int ImgHeight = CVDRVideoFrame::GetImgHeight();

	float* dataCost = new float[boxWidth * boxWidth * m_iDspLevel];
	CDataCostUnit dataCosti;

	std::set<SPoint> EdgePoints;
	pCurrentFrame->GetEdgePoints(EdgePoints, 0);

	std::set<int> boxLabels;
	std::set<SPoint> boxPoints;
	vector<float> dspV;
	std::queue<SPoint> queuePoints;
	std::set<SPoint> setQueuePoints;

	ZImage<bool> EdgeMap(ImgWidth,ImgHeight,1);
	EdgeMap.MakeZero();

	int u, v;
	SPoint pi;

	for(std::set<SPoint>::iterator it = EdgePoints.begin(); it != EdgePoints.end(); it++)
		EdgeMap.at(it->x, it->y, 0) = true;

	for(std::set<SPoint>::iterator it = EdgePoints.begin(); it != EdgePoints.end(); it++){
		//labelImg.at(it->x, it->y, 0) = 0;
		//continue;

		if(EdgeMap.at(it->x, it->y, 0) == false)
			continue;

		boxLabels.clear();
		boxPoints.clear();
		dspV.clear();

		if(!queuePoints.empty()){
			printf("VDR ERROR: Queue is not empty!\n");
			exit(0);
		}

		u = it->x;
		v = it->y;

		//boxing
		int left = u;
		int right = u;
		int up = v;
		int down = v;

		int currentWidth = 1;
		int currentHeight = 1;
		queuePoints.push(*it);
		setQueuePoints.clear();
		setQueuePoints.insert(*it);

		while(!queuePoints.empty()){
			SPoint& head = queuePoints.front();
			boxLabels.insert(labelImg.at(head.x, head.y));
			boxPoints.insert(head);

			for(int neighbori = 0; neighbori< NeighborCount; neighbori++){
				pi.x = head.x + Neighbor[neighbori][0];
				pi.y = head.y + Neighbor[neighbori][1];

				if(pi.x < 0 || pi.x >= ImgWidth || pi.y < 0 || pi.y >= ImgHeight)
					continue;

				if(EdgeMap.at(pi.x, pi.y, 0) == false)
					continue;

				if(setQueuePoints.find(pi) != setQueuePoints.end())
					continue;

				if(pi.x >= left && pi.x <= right && pi.y >= up && pi.y <= down){
					//if(EdgePoints.find(pi) != EdgePoints.end())
					queuePoints.push(pi);
					setQueuePoints.insert(pi);
				}
				else{
					bool inBox = true;
					int li=0, ri=0, ui=0, di=0;
					if(pi.x < left){
						if( left - pi.x + currentWidth > boxWidth )
							inBox = false;
						else
							li = pi.x - left;
					}
					else if(pi.x > right){
						if(pi.x - right + currentWidth > boxWidth)
							inBox = false;
						else
							ri = pi.x - right;
					}

					if(inBox == false)
						continue;

					if(pi.y < up ){
						if(up - pi.y + currentHeight > boxWidth)
							inBox = false;
						else
							ui = pi.y - up;
					}
					else if(pi.y > down){
						if(down - pi.y + currentHeight > boxWidth)
							inBox = false;
						else
							di = pi.y - down;
					}
					if(inBox == true){
						queuePoints.push(pi);
						setQueuePoints.insert(pi);
						left += li;
						right += ri;
						up += ui;
						down += di;
						currentWidth = right - left + 1;
						currentHeight = down - up + 1;
					}
				}
			}
			//setQueuePoints.erase(head);
			queuePoints.pop();
		}

		int DspLevelCount = boxLabels.size();
		if(DspLevelCount <= 1)
			continue;

		dspV.resize(DspLevelCount);
		int loc = 0;
		for(set<int>::iterator labeli = boxLabels.begin();labeli!=boxLabels.end(); labeli++)
			dspV[loc++] = GetDspFromDspLeveli(*labeli);

		//
		int pointIndex = 0;
		for(std::set<SPoint>::iterator iti = boxPoints.begin(); iti != boxPoints.end(); iti++, pointIndex++){
			//	//mark kernal
			int centerX = (left + right) / 2;
			int centerY = (up + down) / 2;
			if((currentWidth < boxWidth && currentHeight < boxWidth) || (iti->x - centerX >= -kenalRadius && iti->x - centerX <= kenalRadius && iti->y - centerY >= -kenalRadius && iti->y - centerY <= kenalRadius))
				EdgeMap.at(iti->x, iti->y, 0) = false;

			dataCosti.Init(dataCost + pointIndex * DspLevelCount, 1, DspLevelCount);
			int bestLabel;
			if(bo == true)
				GetBundleDataCostAt(pCurrentFrame, iti->x, iti->y,dataCosti, NearFrames, bestLabel, dspV, PList);
			else
				GetInitialDataCostAt(pCurrentFrame, iti->x, iti->y,dataCosti, NearFrames, bestLabel, dspV, PList);
			labelImg.SetPixel(iti->x, iti->y, 0, GetLevelFromDspi(dspV[bestLabel], m_iDspLevel));
			pCurrentFrame->SetDspAt(iti->x, iti->y, dspV[bestLabel]);
		}
		//system("pause");

		//test
		//delete [] dataCost;
		//return;

		EdgeGCSmooth(dataCost, labelImg,  vector<int>(boxLabels.begin(), boxLabels.end()), vector<SPoint>(boxPoints.begin(), boxPoints.end()));

		for(std::set<SPoint>::iterator iti = boxPoints.begin(); iti != boxPoints.end(); iti++){
			u = iti->x;
			v = iti->y;
			int labeli = labelImg.at(u, v);
			float dsp =  GetDspFromDspLeveli(labeli);
			pCurrentFrame->SetDspAt(u, v, dsp);
		}
	}
	delete [] dataCost;
}

void CDepthRecoverFuns::EdgeGCSmooth( float* dataCost, ZIntImage& labelImg, std::vector<int>& labels, std::vector<SPoint>& EdgePoints )
{
	int ImgWidth = labelImg.GetWidth();
	int ImgHeight = labelImg.GetHeight();

	int iDspLevelCount = labels.size();
	int numberofPixels = EdgePoints.size();
	float smoothCostLemda = 1;
	float DiscK = CDepthRecoverFuns::GetInstance()->GetTrueDiscK();

	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(numberofPixels, iDspLevelCount);
		gc->setDataCost(dataCost);
		for(int l1 = 0; l1 < iDspLevelCount; l1++)
			for(int l2 = l1; l2 < iDspLevelCount; l2++){
				float cost = min((float)(abs(labels[l1] - labels[l2])), DiscK);
				gc->setSmoothCost(l1, l2, cost);
				gc->setSmoothCost(l2, l1, cost);
			}

			SPoint pi;
			for (int pointIndex = 0; pointIndex <numberofPixels; pointIndex++ ){    // this nested loop sets up a full neighborhood system
				for(int neighbori = 0; neighbori< NeighborCount; neighbori++)
				{
					pi.x = EdgePoints[pointIndex].x + Neighbor[neighbori][0];
					pi.y = EdgePoints[pointIndex].y + Neighbor[neighbori][1];

					if(pi.x < 0 || pi.x >= ImgWidth || pi.y < 0 || pi.y >= ImgHeight)
						continue;

					std::vector<SPoint>::iterator it = lower_bound(EdgePoints.begin(), EdgePoints.end(), pi);

					if(it!= EdgePoints.end() && it->x == pi.x && it->y == pi.y){
						if( pointIndex < it - EdgePoints.begin() )
							gc->setNeighbors(pointIndex, it - EdgePoints.begin(), 1);
					}
					else{
						for(int li = 0; li < iDspLevelCount; li++)
							dataCost[pointIndex * iDspLevelCount + li] += smoothCostLemda * min((float)(abs(labels[li] - labelImg.at(pi.x, pi.y))), DiscK);
					}
				}
			}

			// now set up a grid neighborhood system
			//printf("Before optimization energy is %f\n",gc->compute_energy());
			gc->expansion( 3 );// run expansion for 3 iterations. For swap use gc->swap(num_iterations);
			//printf("\nAfter optimization energy is %f",gc->compute_energy());

			for (int pix =0; pix < numberofPixels; pix++ )
				labelImg.at(EdgePoints[pix].x, EdgePoints[pix].y) = labels[gc->whatLabel(pix)];

			delete gc;
	}
	catch (GCException e){
		e.Report();
	}
}
void CDepthRecoverFuns::EdgeSmooth( CDataCost& DataCost, CDataCost& uMsg, CDataCost& dMsg, CDataCost& lMsg, CDataCost& rMsg, ZIntImage& labelImg,
								   int left, int right, int up, int down, int dspLevelCount, int IterCount, std::set<int>& labels)
{
	int winWidth = right - left + 1;
	int winHeight = down - up + 1;

	int ImgWidth = CVDRVideoFrame::GetImgWidth();
	int ImgHeight = CVDRVideoFrame::GetImgHeight();

	float DiscK = GetTrueDiscK();

	int x, y, px, py;

	for(y = up; y<=down; y++){
		x = left;
		for(int neighbori = 0; neighbori< NeighborCount; neighbori++)
		{
			px = x + Neighbor[neighbori][0];
			py = y + Neighbor[neighbori][1];

			if(px >= left && px<= right && py >= up && py <= down)
				continue;
			if(px >= 0 && px<ImgWidth && py>=0 && py<ImgHeight)
			{
				std::set<int>::iterator it = labels.begin();
				for(int di=0; di<dspLevelCount; di++, it++)
					DataCost.At(x-left, y-up, di) += min((float)(abs(*it - labelImg.at(px, py))), DiscK);
			}
		}
		x = right;
		for(int neighbori = 0; neighbori< NeighborCount; neighbori++)
		{
			px = x + Neighbor[neighbori][0];
			py = y + Neighbor[neighbori][1];

			if(px >= left && px<= right && py >= up && py <= down)
				continue;
			if(px >= 0 && px<ImgWidth && py>=0 && py<ImgHeight)
			{
				std::set<int>::iterator it = labels.begin();
				for(int di=0; di<dspLevelCount; di++, it++)
					DataCost.At(x-left, y-up, di) += min((float)(abs(*it - labelImg.at(px, py))), DiscK);
			}
		}
	}
	for(x = left+1; x<=right-1; x++){
		y = up;
		for(int neighbori = 0; neighbori< NeighborCount; neighbori++)
		{
			px = x + Neighbor[neighbori][0];
			py = y + Neighbor[neighbori][1];

			if(px >= left && px<= right && py >= up && py <= down)
				continue;
			if(px >= 0 && px<ImgWidth && py>=0 && py<ImgHeight)
			{
				std::set<int>::iterator it = labels.begin();
				for(int di=0; di<dspLevelCount; di++, it++)
					DataCost.At(x-left, y-up, di) += min((float)(abs(*it - labelImg.at(px, py))), DiscK);
			}
		}
		y = down;
		for(int neighbori = 0; neighbori< NeighborCount; neighbori++)
		{
			px = x + Neighbor[neighbori][0];
			py = y + Neighbor[neighbori][1];

			if(px >= left && px<= right && py >= up && py <= down)
				continue;
			if(px >= 0 && px<ImgWidth && py>=0 && py<ImgHeight)
			{
				std::set<int>::iterator it = labels.begin();
				for(int di=0; di<dspLevelCount; di++, it++)
					DataCost.At(x-left, y-up, di) += min((float)(abs(*it - labelImg.at(px, py))), DiscK);
			}
		}
	}

	// CBeliefPropagation
	//translateMessage(u, d, l, r, datacost, widthi, heighti);
	CDataCostUnit dataCostUnit, MsgUnit1, MsgUnit2, MsgUnit3, MsgUnit4;
	for (int t = 0; t < IterCount; t++) {
		for (int y = 1; y < winHeight-1; y++){
			for (int x = (y + t + 1) % 2 + 1; x < winWidth-1; x+=2)
			{
				int offset = ( y * winWidth + x ) * dspLevelCount;

				DataCost.GetDataCostUnit(x, y, dataCostUnit);

				//msg(imRef(u, x, y+1),imRef(l, x+1, y),imRef(r, x-1, y), imRef(data, x, y), imRef(u, x, y));
				uMsg.GetDataCostUnit(x, y+1, MsgUnit1);
				lMsg.GetDataCostUnit(x+1, y, MsgUnit2);
				rMsg.GetDataCostUnit(x-1, y, MsgUnit3);
				uMsg.GetDataCostUnit(x, y, MsgUnit4);
				EdgeUpdateMessage(MsgUnit1, MsgUnit2 ,MsgUnit3, dataCostUnit, MsgUnit4, dspLevelCount, labels);

				//msg(imRef(d, x, y-1),imRef(l, x+1, y),imRef(r, x-1, y), imRef(data, x, y), imRef(d, x, y));
				dMsg.GetDataCostUnit(x, y-1, MsgUnit1);
				lMsg.GetDataCostUnit(x+1, y, MsgUnit2);
				rMsg.GetDataCostUnit(x-1, y, MsgUnit3);
				dMsg.GetDataCostUnit(x, y, MsgUnit4);
				EdgeUpdateMessage(MsgUnit1, MsgUnit2 ,MsgUnit3, dataCostUnit, MsgUnit4, dspLevelCount, labels);

				//msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(r, x-1, y), imRef(data, x, y), imRef(r, x, y));
				uMsg.GetDataCostUnit(x, y+1, MsgUnit1);
				dMsg.GetDataCostUnit(x, y-1, MsgUnit2);
				rMsg.GetDataCostUnit(x-1, y, MsgUnit3);
				rMsg.GetDataCostUnit(x, y, MsgUnit4);
				EdgeUpdateMessage(MsgUnit1, MsgUnit2 ,MsgUnit3, dataCostUnit, MsgUnit4, dspLevelCount, labels);

				//msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(l, x+1, y), imRef(data, x, y), imRef(l, x, y));
				uMsg.GetDataCostUnit(x, y+1, MsgUnit1);
				dMsg.GetDataCostUnit(x, y-1, MsgUnit2);
				lMsg.GetDataCostUnit(x+1, y, MsgUnit3);
				lMsg.GetDataCostUnit(x, y, MsgUnit4);
				EdgeUpdateMessage(MsgUnit1, MsgUnit2 ,MsgUnit3, dataCostUnit, MsgUnit4, dspLevelCount, labels);
			}
		}
	}

	//GetDepth(u[0], d[0], l[0], r[0], datacost[0], LabelImg, blockWidth,  blockHeight, TrueX1, TrueY1, TrueX2, TrueY2);
	for (int y = 1; y < winHeight-1; y++) {
		for (int x = 1; x < winWidth-1; x++) {
			// keep track of best value for current pixel
			int bestD = 0;
			float bestVal = 1e20F;

			int offset = ( y * winWidth + x ) * dspLevelCount;

			for (int di = 0; di < dspLevelCount; di++) {
				float val = uMsg.GetValueAt(x, y+1, di) +
					//(u + offset + winWidth * dspLevelCount)[di] +
					dMsg.GetValueAt(x, y-1, di)+
					//(d + offset - winWidth * dspLevelCount)[di] +
					lMsg.GetValueAt(x+1, y, di)+
					//(l + offset + dspLevelCount)[di] +
					rMsg.GetValueAt(x-1, y, di)+
					//r + offset - dspLevelCount)[di] +
					DataCost.GetValueAt(x, y, di); //dataCost + offset)[di];
				if (val < bestVal) {
					bestVal = val;
					bestD = di;
				}
			}
			labelImg.SetPixel( left + x, up + y, 0, bestD);
		}
	}

	//Boundary
	//for(int y = 0; y < winHeight; y++){
	//	labelImg.at( left + 0, up + y) = labelImg.at( left + 1, up + y);
	//	labelImg.at( left + winWidth-1, up + y) = labelImg.at( left + winWidth-2, up + y);
	//}
	//for(int x=0; x<winWidth; x++){
	//	labelImg.at( left + x, up + 0) = labelImg.at( left + x, up + 1);
	//	labelImg.at( left + x, up + winHeight-1) = labelImg.at( left + x, up + winHeight-2);
	//}
}

void CDepthRecoverFuns::EdgeUpdateMessage( CDataCostUnit& srcMsg1, CDataCostUnit& srcMsg2, CDataCostUnit& srcMsg3, CDataCostUnit& dataCost, CDataCostUnit& dstMsg, int dspLevelCount, std::set<int>& labels )
{
	//aggregate and find min
	float minH = 1e20F;
	for (int di = 0; di < dspLevelCount; di++)
	{
		dstMsg[di] = srcMsg1[di] + srcMsg2[di] + srcMsg3[di] + dataCost[di];
		if (dstMsg[di] < minH)
			minH = dstMsg[di];
	}

	// dt
	std::set<int>::iterator pre, it;
	pre = it = labels.begin();
	it++;
	for (int di = 1; di < dspLevelCount; di++, it++) {
		float prev = dstMsg[di-1] + *it - *pre;
		if (prev < dstMsg[di])
			dstMsg[di] = prev;
	}
	pre = it = labels.end();
	it--;
	it--;
	pre--;
	for (int di = dspLevelCount-2; di >= 0; di--, it--) {
		float prev = dstMsg[di+1] + *pre - *it;
		if (prev < dstMsg[di])
			dstMsg[di] = prev;
	}

	// truncate
	minH += GetTrueDiscK();
	for (int di = 0; di < dspLevelCount; di++)
		if (minH < dstMsg[di])
			dstMsg[di] = minH;

	// normalize
	float val = 0;
	for (int di = 0; di < dspLevelCount; di++)
		val += dstMsg[di];

	val /= dspLevelCount;
	for (int di = 0; di < dspLevelCount; di++)
		dstMsg[di] -= val;
}