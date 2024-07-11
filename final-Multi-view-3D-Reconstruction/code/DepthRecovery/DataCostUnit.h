#pragma once

class CDataCostUnit
{
public:
	CDataCostUnit(void);

	~CDataCostUnit(void);

	void Init(float* data, int step, int size);

	float& operator [](int index);

private:
	float * m_pfData;
	int m_iStep;
	int m_iSize;
};
