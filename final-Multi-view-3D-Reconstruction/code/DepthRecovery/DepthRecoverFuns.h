#pragma once
#include "VDRVideoFrame.h"
#include "ZImageUtil.h"
#include <vector>
#include "ImgPartition.h"
#include "DataCost.h"
#include "DataCostUnit.h"
#include <driver_types.h>
#include <string>
#include <sstream>
#include <WmlVector4.h>
#include"RandomForestSingleAttribute.h"
#include"FrameSelectorBase.h"
//#include<WinDef.h>
//#define MAX_PATH 256
//#include<Windows.h>
using std::istringstream ;
using std::string;
class CFrameSelectorBase;
class CDepthRecoverFuns
{
public:

	enum RUNTYPE{
		RUNALL,
		RUNINIT,
		RUNBO,
		RUNDEPTHEXPANSION,
		RUNBODE,
		RUNDSPLITSQUENCE
	};

	CDepthRecoverFuns(void);

	~CDepthRecoverFuns(void);

private:
	void GetPList(std::vector<Wml::Matrix4d>& PList, CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& nearFrames);

	//void GetPListAndPInvList(std::vector<Wml::Matrix4d>& PList, std::vector<Wml::Matrix4d>& PInvList, CVideoFrame* pCurrentFrame, std::vector<CVideoFrame*>& nearFrames);

	void GetInitialDataCost(std::vector<CVDRVideoFrame*>& nearFrames, std::vector<Wml::Matrix4d>& PList,CVDRVideoFrame* pCurrentFrame, CDataCost& outDataCost, ZIntImage& labelImg);

	void GetBundleDataCost( std::vector<CVDRVideoFrame*>& nearFrames, std::vector<Wml::Matrix4d>& PList,CVDRVideoFrame* pCurrentFrame, CDataCost& outDataCost, ZIntImage& labelImg);

	void RefineDspByBP(CDataCost& DataCost, ZIntImage& labelImg, bool addEdgeInfo, ZIntImage* offsetImg = NULL);

	////debug
	//void RefineDspByGpuBp2(CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr, ZIntImage& labelImg, void* offsetImg, void* d_labelImg, size_t d_pitchBytes, bool addEdgeInfo);

	void RefineDspByGpuBp(CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr, ZIntImage& labelImg, void* offsetImg, void* d_labelImg, size_t d_pitchBytes, bool addEdgeInfo);

	void RefineDspBySegm(CVDRVideoFrame* currentFrame, ZIntImage& labelImg, CDataCost& DataCost);

	void _SuperResolutionRefine(CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& nearFrames,int iterCount,
		std::vector<Wml::Matrix4d>& PList, ZIntImage& offsetImg, ZIntImage& labelImg);

	void _SuperResolutionRefineByGPU(int frameCount,int iterCount, CDataCost& subDataCost, cudaPitchedPtr& subDataCostPitchPtr, void* offsetImg,
		ZIntImage& labelImg, void* tmpLableImg, size_t labeImgPitchBytes, size_t ColorImgPitchBytes, size_t DspImgPitchBytes);

	void EdgeDepthInit(CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& NearFrames, std::vector<Wml::Matrix4d>& PList,ZIntImage& labelImg);

	void EdgeDepthGCInit( CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& NearFrames, std::vector<Wml::Matrix4d>& PList, ZIntImage& labelImg);

	void EdgeDepthRefine( CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& NearFrames, std::vector<Wml::Matrix4d>& PList, ZIntImage& labelImg);

	void EdgeSmooth(CDataCost& DataCost, CDataCost& uMsg, CDataCost& dMsg, CDataCost& lMsg, CDataCost& rMsg,
		ZIntImage& labelImg, int left, int right, int up, int down, int dspLevelCount, int IterCount, std::set<int>& labels);

	void EdgeUpdateMessage(CDataCostUnit& srcMsg1, CDataCostUnit& srcMsg2, CDataCostUnit& srcMsg3, CDataCostUnit& dataCost, CDataCostUnit& dstMsg, int dspLevelCount, std::set<int>& labels);

	void EdgeGCSmooth(float* dataCost, ZIntImage& labelImg,std::vector<int>& labels, std::vector<SPoint>& EdgePoints);

	void EdgeGCRefine(CVDRVideoFrame* pCurrentFrame, std::vector<CVDRVideoFrame*>& NearFrames, std::vector<Wml::Matrix4d>& PList, ZIntImage& labelImg, bool bo);

public:
	void EstimateDepth( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost);

	void EstimateDepthByGpu( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr);

	void RefineOneSegm(ZIntImage& labelImg, ZIntImage& segMap, std::vector< Wml::Vector2<int> >* pSegmPoints, CDataCost& dataCost,CVDRVideoFrame* pCurrentFrame);

	void RefineDepth( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost);
	void RefineDepthPF( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost);

	void RefineDepthByGpu( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame,
		CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr, CDataCost& subDataCost, cudaPitchedPtr& subDataCostPitchPtr);

	void RefineDepthWithPfByGpu( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame,
		CDataCost& DataCost, cudaPitchedPtr& dataCostPitchedPtr, CDataCost& subDataCost, cudaPitchedPtr& subDataCostPitchPtr);
	//根据颜色进行了优化
	void UpSampling(std::vector<CVDRVideoFrame*>& FwFrames,	std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame);

	//不使用颜色优化，只插值
	void UpSampling(CVDRVideoFrame* pCurrentFrame,float scale=1.0);

	void RefineEdgeDepth(std::vector<CVDRVideoFrame*>& FwFrames,	std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame);

	void SuperResolutionRefine(std::vector<CVDRVideoFrame*>& FwFrames,	std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame);

	void SuperResolutionRefineByGpu( std::vector<CVDRVideoFrame*>& FwFrames, std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& subDataCost, cudaPitchedPtr& subDataCostPitchPtr);

	void GetNearFrameIndex(std::vector<int>& FwFrameIndex, std::vector<int>& BwFrameIndex,int CurrentIndex, int StartIndex, int EndIndex);

	void GetInitialDataCostAt(CVDRVideoFrame* pCurrentFrame,int u, int v, CDataCostUnit& dataCosti, std::vector<CVDRVideoFrame*>& NearFrames,
		int& bestLabel, std::vector<float>& dspV, std::vector<Wml::Matrix4d>& PList);

	void GetInitialDataCostAt(CVDRVideoFrame* pCurrentFrame,int u, int v, CDataCostUnit& dataCosti,
		std::vector<CVDRVideoFrame*>& NearFrames, int& bestLabel, std::vector<Wml::Matrix4d>& PList);

	void GetBundleDataCostAt(CVDRVideoFrame* pCurrentFrame,int u, int v,
		CDataCostUnit& dataCost, std::vector<CVDRVideoFrame*>& NearFrames, int& bestLabel,
		std::vector<float>& dspV,std::vector<Wml::Matrix4d>& PList);

	double GetDspFromDspLeveli(int DspLeveli){
		return m_fDspV[DspLeveli];
	}

	int GetLevelFromDspi(double dspi, int LevelCount){
		return (LevelCount - 1) * (dspi - m_dDspMin) / (m_dDspMax - m_dDspMin) + 0.5;
	}

	static CDepthRecoverFuns* GetInstance(){
		return &m_Instance;
	}

	double GetDspAtLevelI(double leveli)
	{
		return m_dDspMin * (m_iDspLevel - 1 - leveli)/(m_iDspLevel - 1)  + m_dDspMax * leveli/(m_iDspLevel - 1) ;
	}

public:
	CImgPartition m_ImgPartition;

	string m_pathName;
	//============Block states =============
private:
	int m_iBlockIdx, m_iBlockIdy;
	int m_iOffsetX, m_iOffsetY;
	int m_iBlockWidth, m_iBlockHeight;
	int m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2;

public:
	void SetBlockState(int trueX1, int trueY1, int trueX2, int trueY2, int offsetX, int offsetY, int BlockWidth,int BlockHeight, int blockIdX, int blockIdY){
		m_iTrueX1 = trueX1;
		m_iTrueY1 = trueY1;
		m_iTrueX2 = trueX2;
		m_iTrueY2 = trueY2;
		m_iBlockWidth = BlockWidth;
		m_iBlockHeight = BlockHeight;
		m_iOffsetX = offsetX;
		m_iOffsetY = offsetY;
		m_iBlockIdx = blockIdX;
		m_iBlockIdy = blockIdY;
	}
	void PrintBlockState(){
		printf("BlcokIndex:(%d,%d),OffsetX:%d, OffsetY:%d,Width:%d, Height:%d, LabelRegion:(%d,%d,%d,%d)\n",
			m_iBlockIdx, m_iBlockIdy, m_iOffsetX, m_iOffsetY, m_iBlockWidth, m_iBlockHeight, m_iTrueX1, m_iTrueY1, m_iTrueX2, m_iTrueY2);
	}

	bool IfInTrueRegion(int XinBlock, int YinBlock){
		if(XinBlock + m_iOffsetX >= m_iTrueX1 && XinBlock + m_iOffsetX <= m_iTrueX2 && YinBlock + m_iOffsetY >= m_iTrueY1 && YinBlock + m_iOffsetY <= m_iTrueY2)
			return true;
		return false;
	}

	void GetGlobalCoordinateFrmBlockCoordinate(int XinBlock, int YinBlock, int& GlobalX, int& GlobalY ){
		GlobalX = m_iOffsetX + XinBlock;
		GlobalY = m_iOffsetY + YinBlock;
	}
	//void GetGlobalCoordinateFrmTrueCoordinate(int XinTure, int YinTrue, int& GlobalX, int& GlobalY ){
	//	GlobalX = m_iTrueX1 + XinTure;
	//	GlobalY = m_iTrueY1 + YinTrue;
	//}
	//void GetBlockCoordinateFrmTrueCoordinate(int XinTure, int YinTrue, int& XinBlock, int& YinBlock ){
	//	XinBlock = XinTure + m_iTrueX1 - m_iOffsetX;
	//	YinBlock = YinTrue + m_iTrueY1 - m_iOffsetY;
	//}

private:
	string m_sProjectFilePath;

	float m_fReduceScale;

	bool m_ifGenerateTempResult;
public:
	int m_iStartDst;
	int m_iInitStep;
	int m_iInitFrames;
	int m_iNormalStep;
	int m_iMaxFrames;
	int  m_iSplitOverlap;
	int m_iSplitNo;
private:
	double m_dDspMin;
	double m_dDspMax;
	int m_iDspLevel;
	std::vector<float> m_fDspV;
	bool m_bAutoEstimateDsp;
	bool m_bAutoEstimateRef;
	//Init DataCost
	//float m_fSegmaC;
	//float m_fMissPenalty;//the default color difference when the projecting point is out of the region
	float m_fDataCostWeight;

	int m_iRefinePass;

	//Bundle DataCost
	double m_dColorSigma;
	//double m_dPrjSigma;
	double m_dDspSigma;
	float m_fColorMissPenalty;

	//BP  and  LM
	float m_fDiscK;

	float m_fSegErrRateThreshold;
	float m_fSegSpatial;
	float m_fSegColor;
	float m_fSegMinsize;
	int m_fPlaneFittingSize;

	//==================================================
	int m_iSubIter;
	int m_iSubSample;
	double m_dMinDspSigma;

	//double m_dMinPrjSigma;
	//==================================================
	//Threads and GPU
	bool m_bUseGpu;
	bool m_bUseMI;
	int m_iCpuThreads;
	//==============================================
	RUNTYPE m_RunType;
	public:
		bool m_bUseSC;
		string m_sRandomForrestFilePath;
		bool m_bUsePriorPlane;
		int m_iPlaneNum;
		Wml::Vector4d* m_PlanArr;
		ZIntImage m_segm;
private:
	static CDepthRecoverFuns m_Instance;

public:

	void SetProjectFilePath(istringstream &in);
	void SetProjectFilePath(const string& sProjectFilePath);
	string GetProjectFilePath(){
		return m_sProjectFilePath;
	}
	void CDepthRecoverFuns::SetUseSkyClassifier(istringstream &in){
		in >>m_bUseSC>> m_sRandomForrestFilePath;
		TCHAR FullPath[MAX_PATH];
		TCHAR PrjDir[MAX_PATH];
		GetFullPathName(m_sRandomForrestFilePath.c_str(), MAX_PATH, FullPath, (TCHAR**)&PrjDir);
		m_sRandomForrestFilePath = FullPath;
	}

	void SetPriorPlanes(istringstream &in){
	//	int planNum;
		in>>m_bUsePriorPlane;
		if(m_bUsePriorPlane)
		{
			//in>>m_iPlaneNum;
			std::string planefile;
			in>>planefile;
			std::ifstream fin(planefile);
			string str;
			if(!fin.good()){
			 printf("ERROR: config failed！\n");
			return ;
		    }
			getline(fin, str);
			istringstream num(str);
			num>>m_iPlaneNum;
			m_PlanArr=new Wml::Vector4d[m_iPlaneNum];
			for(int i=0;i<m_iPlaneNum;i++)
			{
				getline(fin, str);
				istringstream num(str);
				num>>m_PlanArr[i][0]>>m_PlanArr[i][1]>>m_PlanArr[i][2]>>m_PlanArr[i][3];
		    }
// 			for(int i=0;i<m_iPlaneNum;i++)
// 			{
// 				std::cout<<m_PlanArr[i][0]<<"  "<<m_PlanArr[i][1]<<"  "<<m_PlanArr[i][2]<<"  "<<m_PlanArr[i][3]<<std::endl;
// 			}
	   }
	}

	void SetReduceScale(istringstream &in){
		in >> m_fReduceScale;
		CVDRVideoFrame::SetScaleVale(m_fReduceScale);
	}
	void SetReduceScale(float fReduceScale){
		m_fReduceScale = fReduceScale;
		CVDRVideoFrame::SetScaleVale(m_fReduceScale);
	}
	float GetReduceScale(){
		return m_fReduceScale;
	}

	void SetIfGenerateTempResult(istringstream &in){
		in >> m_ifGenerateTempResult;
	}
	void SetIfGenerateTempResult(bool ifGenerateTempResult){
		m_ifGenerateTempResult = ifGenerateTempResult;
	}
	bool getIfGenerateTempResult(){
		return m_ifGenerateTempResult;
	}

	void SetStep(istringstream &in){
		in >> m_iStartDst >> m_iInitStep >> m_iInitFrames >> m_iNormalStep >> m_iMaxFrames;
	}
	void SetStep(int iStartDst,int iInitStep,int iInitFrames,int iNormalStep,int iMaxFrames){
		m_iStartDst = iStartDst;
		m_iInitStep = iInitStep;
		m_iInitFrames = iInitFrames;
		m_iNormalStep = iNormalStep;
		m_iMaxFrames = iMaxFrames;
	}
	void GetStep(int& iStartDst,int& iInitStep,int& iInitFrames,int& iNormalStep,int& iMaxFrames){
		iStartDst = m_iStartDst;
		iInitStep = m_iInitStep;
		iInitFrames = m_iInitFrames;
		iNormalStep = m_iNormalStep;
		iMaxFrames = m_iMaxFrames;
	}

	void SetDspPara(istringstream &in){
		m_dDspMin = 1.0e-7;

		in >> m_iDspLevel >> m_dDspMin >> m_dDspMax;
		m_fDspV.resize(m_iDspLevel);

		if(m_dDspMax != 0){
			int layercount = m_iDspLevel - 1;
			if(layercount == 0)
				return;
			for(int i=0;i<m_iDspLevel;i++)
				m_fDspV[i] = m_dDspMin * (layercount-i)/layercount + m_dDspMax * (i)/layercount;
		}
	}
	void SetDspPara(int iDspLevel, double dDspMin, double dDspMax){
		m_iDspLevel = iDspLevel;
		m_dDspMin = dDspMin;
		m_dDspMax = dDspMax;

		if(m_dDspMax != 0){
			int layercount = m_iDspLevel - 1;
			if(layercount == 0)
				return;
			m_fDspV.resize(iDspLevel);
			for(int i=0;i<m_iDspLevel;i++)
				m_fDspV[i] = m_dDspMin * (layercount-i)/layercount + m_dDspMax * (i)/layercount;
		}
	}
	void GetDspPara(int& iDspLevel, double& dDspMin, double& dDspMax){
		iDspLevel = m_iDspLevel;
		dDspMin = m_dDspMin;
		dDspMax = m_dDspMax;
	}
	int GetDspLevel(){
		return m_iDspLevel;
	}

	void SetDspMax(double dspMax){
		m_dDspMax = dspMax;
		int layercount = m_iDspLevel - 1;
		if(layercount > 0){
			for(int i=0;i<m_iDspLevel;i++)
				m_fDspV[i] = m_dDspMin * (layercount-i)/layercount + m_dDspMax * (i)/layercount;
		}
	}

	void SetAutoEstimateDsp(istringstream &in){
		in >> m_bAutoEstimateDsp;
	}
	void SetAutoEstimateRef(istringstream &in){
		in >> m_bAutoEstimateRef;
	}
	void SetAutoEstimateDsp(bool bAutoEstimateDsp){
		m_bAutoEstimateDsp = bAutoEstimateDsp;
	}
	bool GetAutoEstimateDsp(){
		return m_bAutoEstimateDsp;
	}
	bool GetAutoEstimateRef(){
		return m_bAutoEstimateRef;
	}
	void SetDataCostWeight(istringstream &in){
		//in >> m_fSegmaC >> m_fMissPenalty >> m_fDataCostWeight;
		in >> m_fDataCostWeight;
	}
	void SetDataCostWeight(float fDataCostWeight){
		m_fDataCostWeight = fDataCostWeight;
	}
	float GetDataCostWeight(){
		return m_fDataCostWeight;
	}
	float GetTrueDataCostWeight(){
		return (m_iDspLevel - 1) * m_fDataCostWeight / 100.0;
	}

	void SetBundleOptimizationPara(istringstream &in){
		in >> m_dColorSigma >> m_dDspSigma >> m_fColorMissPenalty;
	}
	void SetBundleOptimizationPara(double dColorSigma, double dDspSigma,float fColorMissPenalty){
		m_dColorSigma = dColorSigma;
		m_dDspSigma = dDspSigma;
		m_fColorMissPenalty = fColorMissPenalty;
	}
	void GetBundleOptimizationPara(double& dColorSigma, double& dDspSigma,float& fColorMissPenalty){
		dColorSigma = m_dColorSigma;
		dDspSigma = m_dDspSigma;
		fColorMissPenalty = m_fColorMissPenalty;
	}

	void SetDepthExpansionPara(istringstream &in){
		//in >> m_iSubIter >> m_iSubSample >> m_dMinPrjSigma;
		in >> m_iSubSample >> m_dMinDspSigma; //写错了，写成m_dDspMin，调一天！！！
	}
	void SetDepthExpansionPara(int iSubSample, double dMinDspSigma){
		m_iSubSample = iSubSample;
		m_dMinDspSigma = dMinDspSigma;
	}
	void GetDepthExpansionPara(int& iSubSample, double& dMinDspSigma){
		iSubSample = m_iSubSample;
		dMinDspSigma = m_dMinDspSigma;
	}
	int GetSubSampleCount(){
		return m_iSubSample;
	}

	void SetDiscK(istringstream &in){
		in >> m_fDiscK;
	}
	void SetDiscK(float fDisck){
		m_fDiscK = fDisck;
	}
	float GetDiscK(){
		return m_fDiscK;
	}
	float GetTrueDiscK(){
		return m_fDiscK / 100.0 * (m_iDspLevel - 1);
	}

	void SetBlockPara(istringstream &in){
		int blockCountx, blockCounty;
		float overLap;
		in >> blockCountx >> blockCounty >> overLap;
		m_ImgPartition.SetBlocksCount(blockCountx, blockCounty);
		m_ImgPartition.setOverlap(overLap/100.0F);
	}
	void SetBlockPara(int blockCountx, int blockCounty,	float overLap){
		m_ImgPartition.SetBlocksCount(blockCountx, blockCounty);
		m_ImgPartition.setOverlap(overLap/100.0F);
	}
	void GetBlockPara(int& blockCountx, int& blockCounty, float& overLap){
		blockCountx = m_ImgPartition.getBlockCountX();
		blockCounty = m_ImgPartition.getBlockCountY();
		overLap = 100.0F * m_ImgPartition.getOverlap();
	}

	void SetSegErrRateThreshold(istringstream &in){
		in >> m_fSegErrRateThreshold;
	}
	void SetSegErrRateThreshold(float fSegErrRateThreshold){
		m_fSegErrRateThreshold = fSegErrRateThreshold;
	}
	float GetSegErrRateThreshold(){
		return m_fSegErrRateThreshold;
	}
	float GetTruetSegErrRateThreshold(){
		return m_fSegErrRateThreshold * GetTrueDataCostWeight() / GetTrueDiscK();
	}

	void SetSegPara(istringstream &in){
		in >> m_fSegSpatial >> m_fSegColor >> m_fSegMinsize;
	}
	void SetSegPara(float fSegSpatial,float fSegColor, float fSegMinsize){
		m_fSegSpatial = fSegSpatial;
		m_fSegColor = fSegColor;
		m_fSegMinsize = fSegMinsize;
	}
	void GetSegPara(float& fSegSpatial,float& fSegColor, float& fSegMinsize){
		fSegSpatial = m_fSegSpatial;
		fSegColor = m_fSegColor;
		fSegMinsize = m_fSegMinsize;
	}

	void SetPlaneFittingSize(istringstream &in){
		in >> m_fPlaneFittingSize;
	}
	void SetPlaneFittingSize(float fPlaneFittingSize){
		m_fPlaneFittingSize = fPlaneFittingSize;
	}
	float GetPlaneFittingSize(){
		return m_fPlaneFittingSize;
	}

	void SetIfUseGpu(istringstream &in){
		in >> m_bUseGpu;
	}
	void SetIfUseMI(istringstream &in){
		in >> m_bUseMI;
	}
	void SetIfUseGpu(bool value){
		m_bUseGpu = value;
	}
	bool GetIfUseGpu(){
		return m_bUseGpu;
	}
	bool GetIfUseMI(){
		return m_bUseMI;
	}
	void SetCpuThreadsCount(istringstream &in){
		in >> m_iCpuThreads;
	}
	void SetCpuThreadsCount(int iCpuThreads){
		m_iCpuThreads = iCpuThreads;
	}
	int GetCpuThreadsCount(){
		return m_iCpuThreads;
	}

	void SetRunAll(istringstream &in){
		m_RunType = RUNTYPE::RUNALL;
		in >> m_iRefinePass >> m_iSubIter>>m_iBopfStart>>m_iBopfEnd>>m_iStart>>m_iEnd;
	}

	void SetRunInit(istringstream &in){
		m_RunType = RUNTYPE::RUNINIT;
	}
	void SetRunBo(istringstream &in){
		m_RunType = RUNTYPE::RUNBO;
		in >> m_iRefinePass>>m_iBopfStart>>m_iBopfEnd;
	}

	void SetRunBoDe(istringstream &in){
		m_RunType = RUNTYPE::RUNBODE;
		in >> m_iRefinePass>>m_iBopfStart>>m_iBopfEnd>>m_iSubIter;
	}
	void SetRunDepthExpansion(istringstream &in){
		m_RunType = RUNTYPE::RUNDEPTHEXPANSION;
		in >> m_iSubIter;
	}
	void SetRunSplitSequence(istringstream &in){
		m_RunType = RUNTYPE::RUNDSPLITSQUENCE;
		in >> m_iSplitNo/*>>m_iSplitOverlap*/;
	}

	void SetRunType(RUNTYPE runType){
		m_RunType = runType;
	}
	RUNTYPE GetRunType(){
		return m_RunType;
	}

	void SetRefinePass(int iRefinePass){
		m_iRefinePass = iRefinePass;
	}
	int GetRefinePass(){
		return m_iRefinePass;
	}

	void SetSubIter(int iSubIter){
		m_iSubIter = iSubIter;
	}
	int GetSubIter(){
		return m_iSubIter;
	}

	//MI方法新增;
protected:
	std::vector<std::shared_ptr<ZFloatImage> > m_MiCost;
	std::shared_ptr<ZFloatImage> m_LastDsp;
//public:
//	void Run_Init_MI(int start, int end) ;
public:
	bool Run_Init_MI_At( std::vector<CVDRVideoFrame*>& FwFrames,std::vector<CVDRVideoFrame*>& BwFrames, CVDRVideoFrame* pCurrentFrame, CDataCost& DataCost);

	bool GetColorCount(ZIntImage &labelImgInt,std::vector<CVDRVideoFrame*> &nearFrames, CVDRVideoFrame *pCurrentFrame, std::vector<std::shared_ptr<ZFloatImage>> &m_MiCost ,std::vector<double> &dspV, std::vector<Wml::Matrix4d>& PList);

	void GetDataCostMi(std::vector<CVDRVideoFrame*> &nearFrames, CVDRVideoFrame *pCurrentFrame ,std::vector<std::shared_ptr<ZFloatImage>> &MiCost, CDataCost& outDataCost, ZIntImage& labelImg,std::vector<double> &dspV, std::vector<Wml::Matrix4d>& PList);
	//(nearFrames,pCurrentFrame,block, DataCost, labelImgInt, dspV);
	private:
    double m_dColorSigma2;
	public:
	double GetColorSigma()
	{
		return m_dColorSigma2;
	}
	void SetColorSigma(double tsigma)
	{
		m_dColorSigma2=tsigma;
	}
	public:
		void InitLastDsp()
		{
			m_LastDsp=(std::shared_ptr<ZFloatImage>)new ZFloatImage();
			m_LastDsp->CreateAndInit(CVDRVideoFrame::GetImgWidth(), CVDRVideoFrame::GetImgHeight(),1,0);
		}
		void GetInitialDataCostMiAt(CVDRVideoFrame* pCurrentFrame,int u, int v, CDataCostUnit& dataCosti, std::vector<CVDRVideoFrame*>& NearFrames,
			int& bestLabel, std::vector<float>& dspV, std::vector<Wml::Matrix4d>& PList);

		void GetInitialDataCostMiAt(CVDRVideoFrame* pCurrentFrame,int u, int v, CDataCostUnit& dataCosti,
			std::vector<CVDRVideoFrame*>& NearFrames, int& bestLabel, std::vector<Wml::Matrix4d>& PList);
		void SetmdColorSigma(double tsigma){ m_dColorSigma=tsigma;}
		public:
			int m_iBopfStart;
			int m_iBopfEnd;
			int m_iStart;
			int m_iEnd;
      public:
		  ZIntImage m_skyClassifier;
		 // RandomForestSingleAttribute<float> * m_RF;
		  pRF m_RF;

		  //FrameSelector
		  void SetFrameSelector(CFrameSelectorBase *selector){
			  m_pFrameSelector = selector;
		  }
		  CFrameSelectorBase *m_pFrameSelector;
};