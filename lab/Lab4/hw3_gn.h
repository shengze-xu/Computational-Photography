#ifndef HW3_GN_34804D67
#define HW3_GN_34804D67

using namespace std;
using namespace cv;

struct GaussNewtonParams {
	GaussNewtonParams() :
		exact_line_search(false),
		gradient_tolerance(1e-5),
		residual_tolerance(1e-5),
		max_iter(1000),
		verbose(false)
	{}
	bool exact_line_search; // 使用精确线性搜索还是近似线性搜索
	double gradient_tolerance; // 梯度阈值，当前梯度小于这个阈值时停止迭代
	double residual_tolerance; // 余项阈值，当前余项小于这个阈值时停止迭代
	int max_iter; // 最大迭代步数
	bool verbose; // 是否打印每步迭代的信息
};

struct GaussNewtonReport {
	enum StopType {
		STOP_GRAD_TOL,       // 梯度达到阈值
		STOP_RESIDUAL_TOL,   // 余项达到阈值
		STOP_NO_CONVERGE,    // 不收敛
		STOP_NUMERIC_FAILURE // 其它数值错误
	};
	StopType stop_type; // 优化终止的原因
	double n_iter;      // 迭代次数
};

class ResidualFunction {
public:
	virtual int nR() const = 0;
	virtual int nX() const = 0;
	virtual void eval(double *R, double *J, double *X) = 0;
};

class GaussNewtonSolver {
public:
	virtual double solve(
		ResidualFunction *f, // 目标函数
		double *X,           // 输入作为初值，输出作为结果
		GaussNewtonParams param = GaussNewtonParams(), // 优化参数
		GaussNewtonReport *report = nullptr // 优化结果报告
	) = 0;
};

class Solver2721 : public GaussNewtonSolver {
public:
	virtual double solve(
		ResidualFunction *f, // 目标函数
		double *X,           // 输入作为初值，输出作为结果
		GaussNewtonParams param = GaussNewtonParams(), // 优化参数
		GaussNewtonReport *report = nullptr // 优化结果报告
	)
	{
		double *x = X;
		double alpha = 1;
		int nR = f->nR();
		int nX = f->nX();
		double *J = new double[nR*nX];
		double *R = new double[nR];
		double *delta_x = new double[nR];	
		int count;
		for (count = 0; count< param.max_iter; count++) {
			f->eval(R, J, x);
			Mat mat_R(nR, 1, CV_64FC1, R);
			Mat mat_J(nR, nX, CV_64FC1, J);
			Mat delta_x(nX, 1, CV_64FC1);
			cv::solve(mat_J, mat_R, delta_x, DECOMP_SVD);

			double Res = mat_R.at<double>(0, 0);
			double Gra = delta_x.at<double>(0, 0);

			for (int i = 0; i < nR; i++) {
				double r = abs(mat_R.at<double>(i, 0));
				if ( r > Res) {
					Res = r;
				}
			}
			for (int i = 0; i < nX; i++) {
				double g = abs(delta_x.at<double>(i, 0));
				if (g > Gra) {
					Gra = g;
				}
			}

			if (Res <= param.residual_tolerance) {
				report->stop_type = report->STOP_RESIDUAL_TOL;
				report->n_iter = count;
				return 0;
			}
			if (Gra <= param.gradient_tolerance) {
				report->stop_type = report->STOP_GRAD_TOL;
				report->n_iter = count;
				return 0;
			}

			for (int i = 0; i < nX; i++) {
				x[i] += alpha * delta_x.at<double>(i, 0);
			}
		}

		report->stop_type = report->STOP_NO_CONVERGE;
		report->n_iter = count;
		return 1;

	}
};

#endif /* HW3_GN_34804D67 */