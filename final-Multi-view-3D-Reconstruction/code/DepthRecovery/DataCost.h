#pragma once
#include "DataCostUnit.h"
#include "ZImageUtil.h"

class CDataCost
{
public:
	CDataCost();

	CDataCost(int width, int height, int dspLevel, bool SetZero = false);

	void Create(int width, int height, int dspLevel, bool SetZero = false);

	~CDataCost(void);

	float GetValueAt(int x, int y, int leveli);

	float& At(int x, int y, int leveli);

	void GetDataCostUnit(int x, int y, CDataCostUnit& dataCostUnit);

	float* GetBits(){
		return m_pfData;
	}

	int GetWidth(){
		return m_iWidth;
	}

	int GetChannel(){
		return m_iDspLevels;
	}

	void SetLabelImg(int offsetX, int offsetY, int blockWidth, int blockHeight, ZIntImage& labelImg);

private:
	int m_iWidth;
	int m_iHeight;
	int m_iDspLevels;

	float* m_pfData;
	int m_iSlice;
	int m_iTotal;

	//MI
public:
	void SetValueAt(int x, int y, int leveli,double tvalue);
};