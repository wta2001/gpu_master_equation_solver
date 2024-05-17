#ifndef TIANA_SOLVER_H
#define TIANA_SOLVER_H
#include<complex>

namespace NeoSolver {

void solve(std::complex<double>* oper_data, std::complex<double>* psi0_data, std::complex<double>* eops_data,
    double* expect, int q_dim, int t_steps, int n_oper, int n_eops, int ntraj);

}

#endif