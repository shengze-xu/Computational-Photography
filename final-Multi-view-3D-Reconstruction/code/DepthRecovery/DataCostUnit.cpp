#include "DataCostUnit.h"
#include "stdio.h"
#include <stdlib.h>

CDataCostUnit::CDataCostUnit(void)
{
}

CDataCostUnit::~CDataCostUnit(void)
{
}

float& CDataCostUnit::operator[]( int index )
{
	if(index >= m_iSize){
		printf("ERROR:The index of DataCostUnit is out of range!(index:%d)\n", index);
		exit(0);
	}
	return m_pfData[index * m_iStep];
}

void CDataCostUnit::Init( float* data, int step, int size )
{
	m_pfData = data;
	m_iStep = step;
	m_iSize = size;
}
