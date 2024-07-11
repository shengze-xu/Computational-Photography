#pragma once

#include "WmlMatrix3.h"
#include "WmlMatrix4.h"

class CCameraParameters
{
public:
	CCameraParameters(void);

	~CCameraParameters(void);

	void GetDpFrmImgCoordAndPlan(double u, double v,  Wml::Vector4d w_pt4d,double &dsp);

	void GetWorldCoordFrmImgCoord(int u, int v, double desp, Wml::Vector3d& w_pt3d, bool scaled);

	void GetWorldCoordFrmImgCoord(double u, double v, double desp, Wml::Vector3d& w_pt3d, bool scaled);

	void GetImgCoordFrmWorldCoord(float& out_u, float& out_v, float& out_desp, Wml::Vector3d& w_pt3d, bool scaled);

	void GetP(Wml::Matrix4d& P);

	void GetTransposeP(Wml::Matrix4d& TransposeP);

	void InverseExternalPara();

	static void InverseP(Wml::Matrix4d& P);

	void GetCameraCoordFrmImgCoord(int u, int v, double desp, Wml::Vector3d& c_pt3d, bool scaled);

	void GetCameraCoordFrmWorldCoord(Wml::Vector3d& c_pt3d, const Wml::Vector3d& w_pt3d);

public:
	Wml::Matrix3d  m_K; //Intrinsic Matrix
	Wml::Matrix3d  m_Scaled_K; //Scaled Intrinsic Matrix

	Wml::Matrix3d  m_R; //Rotational Matrix
	Wml::Vector3d  m_T;  //Translation Vector
};

inline void CCameraParameters::GetDpFrmImgCoordAndPlan(double u, double v,  Wml::Vector4d plan_abcd_4d,double &dsp)
{
	Wml::Matrix3d RT = m_R;
	Wml::Vector3d O=m_T;
	Wml::Vector3d tm_T=-m_R.Transpose()*m_T;
	//	out_u = m_K(0,0) * c_pt3d[0] / c_pt3d[2] + m_K(0,2);
	//out_v = m_K(1,1) * c_pt3d[1] / c_pt3d[2] + m_K(1,2);
	Wml::Vector3d c_pt3d,tc_pt3d;
	double Xc=(u-m_K(0,2))/m_K(0,0);
	double Yc=(v-m_K(1,2))/m_K(1,1);
	c_pt3d[0]=Xc;c_pt3d[1]=Yc;c_pt3d[2]=1;

	for(int i=0;i<3;i++) tc_pt3d[i]=c_pt3d[i]-tm_T[i];

	for(int i=0; i<3; ++i){
		for(int j=0; j<3; ++j)
			c_pt3d[i] += RT(i,j) * tc_pt3d[j];
	}
	Wml::Vector3d d_vec;
	d_vec[0]=c_pt3d[0]-O[0];
	d_vec[1]=c_pt3d[1]-O[1];
	d_vec[2]=c_pt3d[2]-O[2];

	double a=plan_abcd_4d[0];
	double b=plan_abcd_4d[1];
	double c=plan_abcd_4d[2];
	double d=plan_abcd_4d[3];
	double t=-(d+c*O[2]+b*O[1]+a*O[0])/(a*d_vec[0]+b*d_vec[1]+c*d_vec[2]);
	dsp=t*sqrtl(d_vec[0]*d_vec[0]+d_vec[1]*d_vec[1]+d_vec[2]*d_vec[2]);
}