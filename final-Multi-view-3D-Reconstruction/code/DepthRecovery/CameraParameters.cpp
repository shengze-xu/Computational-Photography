#include "CameraParameters.h"

CCameraParameters::CCameraParameters(void)
{
}

CCameraParameters::~CCameraParameters(void)
{
}

void CCameraParameters::GetWorldCoordFrmImgCoord(int u, int v, double desp, Wml::Vector3d& w_pt3d, bool scaled)
{
	Wml::Vector3d c_pt3d;
	c_pt3d[0] = u;
	c_pt3d[1] = v;
	c_pt3d[2] = 1.0/desp;
	//c_pt3d[2] = 1.0 /( m_fDMin + (m_fDMax - m_fDMin) / m_iDepthLevelCount * DepthLeveli );

	//Calibrate image position, assuming skew = 0
	if(scaled ==false){
		c_pt3d[0] = (c_pt3d[0]-m_K(0,2)) * c_pt3d[2] / m_K(0,0);
		c_pt3d[1] = (c_pt3d[1]-m_K(1,2)) * c_pt3d[2] / m_K(1,1);
	}
	else{
		c_pt3d[0] = (c_pt3d[0]-m_Scaled_K(0,2)) * c_pt3d[2] / m_Scaled_K(0,0);
		c_pt3d[1] = (c_pt3d[1]-m_Scaled_K(1,2)) * c_pt3d[2] / m_Scaled_K(1,1);
	}

	//CameraCoordToWorldCoord
	for(int i=0; i<3; ++i){
		w_pt3d[i] = 0;
		for(int j=0; j<3; ++j)
			w_pt3d[i] += m_R(i,j) * c_pt3d[j];
		w_pt3d[i] += m_T[i];
	}
}
void CCameraParameters::GetCameraCoordFrmImgCoord(int u, int v, double desp, Wml::Vector3d& c_pt3d, bool scaled)
{
	c_pt3d[0] = u;
	c_pt3d[1] = v;
	c_pt3d[2] = 1.0/desp;
	//c_pt3d[2] = 1.0 /( m_fDMin + (m_fDMax - m_fDMin) / m_iDepthLevelCount * DepthLeveli );

	//Calibrate image position, assuming skew = 0
	if(scaled == false){
		c_pt3d[0] = (c_pt3d[0]-m_K(0,2)) * c_pt3d[2] / m_K(0,0);
		c_pt3d[1] = (c_pt3d[1]-m_K(1,2)) * c_pt3d[2] / m_K(1,1);
	}
	else{
		c_pt3d[0] = (c_pt3d[0]-m_Scaled_K(0,2)) * c_pt3d[2] / m_Scaled_K(0,0);
		c_pt3d[1] = (c_pt3d[1]-m_Scaled_K(1,2)) * c_pt3d[2] / m_Scaled_K(1,1);
	}
}

void CCameraParameters::GetWorldCoordFrmImgCoord( double u, double v, double desp, Wml::Vector3d& w_pt3d, bool scaled)
{
	Wml::Vector3d c_pt3d;
	c_pt3d[0] = u;
	c_pt3d[1] = v;
	c_pt3d[2] = 1.0/desp;
	//c_pt3d[2] = 1.0 /( m_fDMin + (m_fDMax - m_fDMin) / m_iDepthLevelCount * DepthLeveli );

	//Calibrate image position, assuming skew = 0
	if(scaled ==false){
		c_pt3d[0] = (c_pt3d[0]-m_K(0,2)) * c_pt3d[2] / m_K(0,0);
		c_pt3d[1] = (c_pt3d[1]-m_K(1,2)) * c_pt3d[2] / m_K(1,1);
	}
	else{
		c_pt3d[0] = (c_pt3d[0]-m_Scaled_K(0,2)) * c_pt3d[2] / m_Scaled_K(0,0);
		c_pt3d[1] = (c_pt3d[1]-m_Scaled_K(1,2)) * c_pt3d[2] / m_Scaled_K(1,1);
	}

	//CameraCoordToWorldCoord
	for(int i=0; i<3; ++i){
		w_pt3d[i] = 0;
		for(int j=0; j<3; ++j)
			w_pt3d[i] += m_R(i,j) * c_pt3d[j];
		w_pt3d[i] += m_T[i];
	}
}

//void CCameraParameters::GetImgCoordFrmWorldCoord(double& out_u, double& out_v, double& out_desp, Wml::Vector3d& w_pt3d, bool scaled)
void CCameraParameters::GetImgCoordFrmWorldCoord(float& out_u, float& out_v, float& out_desp, Wml::Vector3d& w_pt3d, bool scaled)
{
	Wml::Matrix3d RT = m_R.Transpose();
	Wml::Vector3d c_pt3d;
	for(int i=0; i<3; ++i){
		c_pt3d[i] = 0;
		for(int j=0; j<3; ++j)
			c_pt3d[i] += RT(i,j) * (w_pt3d[j]  - m_T[j]);
	}

	if(scaled == false){
		out_u = m_K(0,0) * c_pt3d[0] / c_pt3d[2] + m_K(0,2);
		out_v = m_K(1,1) * c_pt3d[1] / c_pt3d[2] + m_K(1,2);
	}
	else{
		out_u = m_Scaled_K(0,0) * c_pt3d[0] / c_pt3d[2] + m_Scaled_K(0,2);
		out_v = m_Scaled_K(1,1) * c_pt3d[1] / c_pt3d[2] + m_Scaled_K(1,2);
	}
	out_desp = 1.0 / c_pt3d[2];

	//out_DepthLeveli = (1.0 / c_pt3d[2] - m_fDMin) / (m_fDMax - m_fDMin) * m_iDepthLevelCount;
}

void CCameraParameters::GetP( Wml::Matrix4d& P )
{
	for(int i=0 ;i<3 ;i++){
		for(int j=0; j<3; j++){
			P(i, j) = m_R(i, j);
		}
		P(i, 3) = m_T[i];
		P(3, i) = 0.0;
	}
	P(3, 3) = 1.0;
}

void CCameraParameters::GetTransposeP( Wml::Matrix4d& TransposeP )
{
	Wml::Matrix3d TransposeR = m_R.Transpose();
	Wml::Vector3d newT = - TransposeR * m_T;

	for(int i=0 ;i<3 ;i++){
		for(int j=0; j<3; j++){
			TransposeP(i, j) = TransposeR(i, j);
		}
		TransposeP(i, 3) = newT[i];
		TransposeP(3, i) = 0.0;
	}
	TransposeP(3, 3) = 1.0;
}

void CCameraParameters::InverseP( Wml::Matrix4d& P )
{
	Wml::Matrix3d rot;
	Wml::Vector3d T;

	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			rot(i,j) = P(j,i);
		}
		T[i] = P(i, 3);
	}
	T = -rot*T;

	for(int i=0 ;i<3 ;i++){
		for(int j=0; j<3; j++){
			P(i, j) = rot(i, j);
		}
		P(i, 3) = T[i];
		P(3, i) = 0.0;
	}
	P(3, 3) = 1.0;
}

void CCameraParameters::InverseExternalPara()
{
	Wml::Matrix3d rot;
	Wml::Vector3d T;

	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++)
			rot(i,j) = m_R(j,i);
	}

	m_T = -rot * m_T;

	for(int i=0 ;i<3 ;i++){
		for(int j=0; j<3; j++){
			m_R(i, j) = rot(i, j);
		}
	}
}

void CCameraParameters::GetCameraCoordFrmWorldCoord(Wml::Vector3d& c_pt3d, const Wml::Vector3d& w_pt3d)
{
	Wml::Matrix3d  tR; //Rotational Matrix
	Wml::Vector3d  tT;  //Translation Vector
	tR=m_R.Transpose();
	tT=-tR*m_T;
	for(int i=0; i<3; ++i){
		c_pt3d[i] = 0;
		for(int j=0; j<3; ++j)
			c_pt3d[i] += tR(i,j) * w_pt3d[j];
		c_pt3d[i] += tT[i];
	}
}