#pragma once
#include <array>
#include <vector>
#include "WmlVector3.h"

class CTrackPoint{
public:
	int m_iFrameNo;
	int m_iIndex;
	double m_dX;
	double m_dY;
};

class CMatchLinker
{
public:
	CTrackPoint &Point(size_t index){
		if(index < m_ptList.size())
			return m_ptList[index];
		throw std::exception("ERROR:CTrackPoint::index out of range!");
	}
	void Reserve(int iCount){
		m_ptList.resize(iCount);
	}
	size_t Count(){
		return m_ptList.size();
	}

public:
	int m_iFlag;
	int m_iIndex;
	int m_iErrorLevel;
	Wml::Vector3d m_v3D;

protected:
	std::vector<CTrackPoint> m_ptList;
};

class CMatchPoint{
public:
	double m_dX;
	double m_dY;
	std::shared_ptr<CMatchLinker> m_pMatchLinker;
};