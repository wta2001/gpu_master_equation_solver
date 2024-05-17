import qutip
import numpy as np
cimport numpy as cnp
from libcpp.complex cimport complex

cdef extern from "neo_solver.hh" namespace "NeoSolver":
    void solve(complex[double]* oper_data, complex[double]* psi0_data, complex[double]* eops_data,
        double* expect, int q_dim, int t_steps, int n_oper, int n_eops, int ntraj)

cdef class cusolve:
    cdef cnp.ndarray expect_c
    def __cinit__(self, H, psi0, tlist, c_ops, e_ops, ntraj=1000):
        cdef int t_steps = len(tlist)-1
        cdef double dt = (tlist[-1]-tlist[0])/t_steps
        Heff = H - 1j/2*sum(L.dag()*L for L in c_ops)
        oper_list = [1-1j*dt*Heff] + [op*np.sqrt(dt) for op in c_ops]
        cdef cnp.ndarray[cnp.complex128_t, ndim=1] psi0_in = np.ravel(psi0.full()).copy()
        cdef cnp.ndarray[cnp.complex128_t, ndim=1] oper_in = np.concatenate([np.ravel(op.full()) for op in oper_list]).copy()
        cdef cnp.ndarray[cnp.complex128_t, ndim=1] eops_in = np.concatenate([np.ravel(op.full()) for op in e_ops]).copy()
        cdef complex[double]* psi0_data = <complex[double]*> psi0_in.data
        cdef complex[double]* oper_data = <complex[double]*> oper_in.data
        cdef complex[double]* eops_data = <complex[double]*> eops_in.data
        cdef int q_dim = len(psi0_in)
        cdef int n_oper = len(oper_list)
        cdef int n_eops = len(e_ops)
        self.expect_c = np.empty((n_eops, t_steps+1), dtype='float64')
        cdef double* expect_data = <double*> self.expect_c.data
        solve(oper_data, psi0_data, eops_data, expect_data, q_dim, t_steps, n_oper, n_eops, ntraj)
    
    @property
    def expect(self):
        return self.expect_c