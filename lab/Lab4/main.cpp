#include <opencv2/opencv.hpp>
#include "hw3_gn.h"

using namespace cv;
using namespace std;

#define number 753
#define dimension 3

class ObjectFunction : public ResidualFunction {
public:
	double *x, *y, *z;

	ObjectFunction(){
		FILE* file;
		fopen_s(&file, "ellipse753.txt", "r");	
		x = new double[number];
		y = new double[number];
		z = new double[number];
		for (int i = 0; i < number; i++) {
			fscanf_s(file, "%lf", &x[i]);
			fscanf_s(file, "%lf", &y[i]);
			fscanf_s(file, "%lf", &z[i]);
		}
	}

	~ObjectFunction() {
		delete x;
		delete y;
		delete z;
	}

	double Square(double x) {
		return x * x;
	}

	double Cube(double x) {
		return x * x * x;
	}

	double Residual_value(double *eclipse, int i) {
		double A = eclipse[0];
		double B = eclipse[1];
		double C = eclipse[2];
		return 1 - 1.0 / Square(A)*Square(x[i]) - 1.0 / Square(B)*Square(y[i]) - 1.0 / Square(C)*Square(z[i]);
	}	

	virtual int nR() const {
		return number;
	}

	virtual int nX() const {
		return dimension;
	}

	virtual void eval(double *R, double *J, double *eclipse) {
		for (int i = 0; i < number; i++) {
			R[i] = Residual_value(eclipse, i);
			J[i * 3 + 0] = -2 * Square(x[i])/Cube(eclipse[0]);
			J[i * 3 + 1] = -2 * Square(y[i])/Cube(eclipse[1]);
			J[i * 3 + 2] = -2 * Square(z[i])/Cube(eclipse[2]);
		}
	}
};

int main(int argc, char* argv[]) {
	double *Eclipse = new double[3];
	for (int i = 0; i < 3; i++) {
		Eclipse[i] = 1;
	}
	ResidualFunction* f= new ObjectFunction();
	GaussNewtonParams param = GaussNewtonParams();
	GaussNewtonReport report;
	Solver2721* solver = new Solver2721();
	solver->solve(f, Eclipse, param, &report);
	cout << "迭代次数: " << report.n_iter << endl;
	cout << "A:" << " " << Eclipse[0] << endl;
	cout << "B:" << " " << Eclipse[1] << endl;
	cout << "C:" << " " << Eclipse[2] << endl;
	if (report.stop_type == report.STOP_RESIDUAL_TOL) {
		cout << "StopType: 余项达到阈值" << endl;
	}
	else if (report.stop_type == report.STOP_GRAD_TOL) {
		cout << "StopType: 梯度达到阈值" << endl;
	}
	else if (report.stop_type == report.STOP_NO_CONVERGE) {
		cout << "StopType: 不收敛" << endl;
	}
	else {
		cout << "StopType: 其它数值错误" << endl;
	}
}