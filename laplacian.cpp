#include "dlib/matrix.h"
#include "svm.h"

static inline double rbf(const svm_node *x, const svm_node *y, const double gamma) {
    double sum = 0;
    while(x->index != -1 && y->index !=-1)
    {
        if(x->index == y->index)
        {
            double d = x->value - y->value;
            sum += d*d;
            ++x;
            ++y;
        }
        else
        {
            if(x->index > y->index)
            {
                sum += y->value * y->value;
                ++y;
            }
            else
            {
                sum += x->value * x->value;
                ++x;
            }
        }
    }

    while(x->index != -1)
    {
        sum += x->value * x->value;
        ++x;
    }

    while(y->index != -1)
    {
        sum += y->value * y->value;
        ++y;
    }
    return exp(-gamma*sum);
}

int partition(dlib::matrix<double> &arr, int l, int r) { 
	double x = arr(r), i = l; 
    double temp;
	for (int j = l; j <= r - 1; j++) { 
		if (arr(j) >= x) { 
            temp = arr(i);
            arr(i) = arr(j);
            arr(j) = temp;
			i++; 
		} 
	} 
    temp = arr(i);
    arr(i) = arr(r);
    arr(r) = temp;
	return i; 
} 

double kthLargest(dlib::matrix<double> &arr, int l, int r, int k) { 
	if (k > 0 && k <= r - l + 1) { 
		int pos = partition(arr, l, r); 
		if (pos-l == k-1) 
			return arr(pos); 
		if (pos-l > k-1)
			return kthLargest(arr, l, pos-1, k); 
		return kthLargest(arr, pos+1, r, k-pos+l-1); 
	} 
	return INT_MAX; 
} 

void laplacian(const svm_parameter &param, svm_problem &prob, double **PHI_tilde) {
    double gamma = param.lap_gamma;
    int n_neighbors = param.n_neighbors;
    int nu = param.nu_eigen;
    double lambda = param.lmbda;
    double mu = param.mu;
    double p = param.lap_p;

    dlib::matrix<double> W(prob.l, prob.l);
    for (int i = 0; i < prob.l; i++) {
        for (int j = 0; j < i + 1; j++) {
            if (i == j) W(i, j) = 0;
            else {
                W(i, j) = rbf(prob.x[i], prob.x[j], gamma);
                W(j, i) = W(i, j);
            }
        }
    }

    dlib::matrix<double> row;
    double kth;
    for (int i = 0; i < W.nr(); i++) {
        row = dlib::rowm(W, i);
        kth = kthLargest(row, 0, row.nc() - 1, n_neighbors);
        for (int j = 0; j < W.nc(); j++) {
            if (W(i, j) < kth) W(i, j) = 0;
        }
    }

    W = (W + dlib::trans(W)) / 2;

    dlib::matrix<double> D = dlib::sum_cols(W);
    dlib::matrix<double> D_inv = dlib::diagm(dlib::reciprocal(dlib::sqrt(D)));
    dlib::matrix<double> L = D_inv * (dlib::diagm(D) - W) * D_inv;

    dlib::eigenvalue_decomposition<dlib::matrix<double>> eig(L);
    dlib::matrix<double> SIGMA_L = eig.get_real_eigenvalues();
    dlib::matrix<double> PHI = eig.get_pseudo_v();
    dlib::sort_columns(PHI, SIGMA_L);

    SIGMA_L = dlib::rowm(SIGMA_L, dlib::range(0, nu - 1));
    PHI = dlib::colm(PHI, dlib::range(0, nu - 1));

    dlib::matrix<double> LAMBDA(prob.l, 1);
    for (int i = 0; i < prob.l; i++) {
        if (isnan(prob.y[i])) LAMBDA(i) = 0;
        else LAMBDA(i) = lambda;
    }

    dlib::matrix<double> S = dlib::trans(PHI) * dlib::diagm(LAMBDA) * PHI + mu * p * dlib::diagm(SIGMA_L);

    dlib::eigenvalue_decomposition<dlib::matrix<double>> eig_S(S);
    dlib::matrix<double> SIGMA_S = eig_S.get_real_eigenvalues();
    dlib::matrix<double> V = eig_S.get_pseudo_v();

    dlib::matrix<double> PHI_til = PHI * V * dlib::diagm(dlib::reciprocal(dlib::sqrt(SIGMA_S)));

    dlib::matrix<double> y(prob.l, 1);
    for (int i = 0; i < prob.l; i++) {
        if (isnan(prob.y[i])) {
            y(i) = 0;
        } else {
            y(i) = prob.y[i];
        }
    }

    dlib::matrix<double> y_virtual = PHI_til * dlib::trans(PHI_til) * dlib::diagm(LAMBDA) * y;

	for (int i = 0; i < prob.l; i++) {
		prob.y[i] = y_virtual(i);
	}

    for (int i = 0; i < prob.l; i++) {
        for (int j = 0; j < param.nu_eigen; j++) {
            PHI_tilde[i][j] = PHI_til(i, j);
        }
    }
}