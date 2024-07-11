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
	bool exact_line_search; // ʹ�þ�ȷ�����������ǽ�����������
	double gradient_tolerance; // �ݶ���ֵ����ǰ�ݶ�С�������ֵʱֹͣ����
	double residual_tolerance; // ������ֵ����ǰ����С�������ֵʱֹͣ����
	int max_iter; // ����������
	bool verbose; // �Ƿ��ӡÿ����������Ϣ
};

struct GaussNewtonReport {
	enum StopType {
		STOP_GRAD_TOL,       // �ݶȴﵽ��ֵ
		STOP_RESIDUAL_TOL,   // ����ﵽ��ֵ
		STOP_NO_CONVERGE,    // ������
		STOP_NUMERIC_FAILURE // ������ֵ����
	};
	StopType stop_type; // �Ż���ֹ��ԭ��
	double n_iter;      // ��������
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
		ResidualFunction *f, // Ŀ�꺯��
		double *X,           // ������Ϊ��ֵ�������Ϊ���
		GaussNewtonParams param = GaussNewtonParams(), // �Ż�����
		GaussNewtonReport *report = nullptr // �Ż��������
	) = 0;
};

class Solver2721 : public GaussNewtonSolver {
public:
	virtual double solve(
		ResidualFunction *f, // Ŀ�꺯��
		double *X,           // ������Ϊ��ֵ�������Ϊ���
		GaussNewtonParams param = GaussNewtonParams(), // �Ż�����
		GaussNewtonReport *report = nullptr // �Ż��������
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