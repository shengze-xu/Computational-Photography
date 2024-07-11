#include "stdio.h"
#include "windows.h"
#include "DataCost.h"
#include <string.h>
#include <stdlib.h>

CDataCost::CDataCost(int width, int height, int dspLevel, bool SetZero)
{
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDspLevels = 0;
	Create(width, height, dspLevel, SetZero);
}

CDataCost::CDataCost()
{
	m_iWidth = 0;
	m_iHeight = 0;
	m_iDspLevels = 0;

	m_iSlice = 0;
	m_iTotal = 0;
	m_pfData = 0;
}

void CDataCost::Create( int width, int height, int dspLevel, bool SetZero /*= false*/ )
{
	if(m_iWidth == width && m_iHeight == height && m_iDspLevels == dspLevel){
		if(SetZero == true)
			memset(m_pfData, 0, m_iTotal * sizeof(float));
		return;
	}

	m_iWidth = width;
	m_iHeight = height;
	m_iDspLevels = dspLevel;

	m_iSlice = m_iWidth * m_iHeight;
	m_iTotal = m_iSlice * m_iDspLevels;
	m_pfData = new float[m_iTotal];

	if(SetZero == true)
		memset(m_pfData, 0, m_iTotal * sizeof(float));
}

CDataCost::~CDataCost(void)
{
	if(m_iTotal != 0)
		delete [] m_pfData;
}

float CDataCost::GetValueAt( int x, int y, int leveli )
{
	int index = leveli * m_iSlice + y * m_iWidth + x;
	if(index > m_iTotal)
	{
		printf("ERROR:The index of dataCost is out of range!(x:%d,y:%d,leveli:%d)\n", x, y, leveli);
		return -1;
	}
	return m_pfData[index];
}

//MI
void CDataCost::SetValueAt( int x, int y, int leveli,double tvalue )
{
	int index = leveli * m_iSlice + y * m_iWidth + x;
	if(index > m_iTotal)
	{
		printf("ERROR:The index of dataCost is out of range!(x:%d,y:%d,leveli:%d)\n", x, y, leveli);
		return ;
	}
	 m_pfData[index]=tvalue;
}

float& CDataCost::At( int x, int y, int leveli )
{
	int index = leveli * m_iSlice + y * m_iWidth + x;
	if(index > m_iTotal)
	{
		printf("ERROR:The index of dataCost is out of range!(x:%d,y:%d,leveli:%d)\n", x, y, leveli);
		return m_pfData[0];
	}
	return m_pfData[index];
}

void CDataCost::GetDataCostUnit( int x, int y, CDataCostUnit& dataCostUnit )
{
	int index = y * m_iWidth + x;
	if(index >= m_iSlice){
		printf("ERROR:The index of dataCost is out of range!(x:%d,y:%d)\n", x, y);
		exit(0);
	}
	dataCostUnit.Init(m_pfData + index, m_iSlice, m_iDspLevels);
}

void CDataCost::SetLabelImg(int offsetX, int offsetY, int blockWidth, int blockHeight, ZIntImage& labelImg )
{
	for(int j=0; j<blockHeight && j + offsetY < labelImg.GetHeight(); j++)
		for(int i=0; i<blockWidth && i + offsetX < labelImg.GetWidth(); i++){
			float MinValue = 1e20;
			int MinDi = 0;
			for(int di=0; di<m_iDspLevels; di++)
				if(GetValueAt( i, j, di) < MinValue){
					MinValue = GetValueAt( i, j, di);
					MinDi = di;
				}
			labelImg.at(offsetX + i, offsetY + j) = MinDi;
		}
}