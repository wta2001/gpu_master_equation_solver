#include<cuda_runtime.h>
#include<cuda/std/complex>
#include<curand_kernel.h>
#include <complex>

#define P(A, B) idx*(n_oper+1)*q_dim+(A)*q_dim+B
#define Q(A, B) idx*n_eops*q_dim+(A)*q_dim+B

namespace NeoSolver {

using complex = std::complex<double>;
using cu_complex = cuda::std::complex<double>;

__global__ void CUDAsolve(cu_complex* oper, cu_complex* psi0, cu_complex* psi, cu_complex* psi_eops, cu_complex* eops,
    curandState_t* state, double* expect, double* product, double* cumulate_weight,
    int* flag, int q_dim, int t_steps, int n_oper, int n_eops, int ntraj) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = threadIdx.y;
    
    if(idx<ntraj&&idy==0) {
        curand_init(idx, 0, 0, &state[idx]);
    }
    if(idx<ntraj&&idy<q_dim) {
        psi[P(n_oper, idy)] = psi0[idy];
    }
    __syncthreads();

    for(int time=0; time<t_steps; time++) {
        if(idx<ntraj&&idy<q_dim) {
            for(int j=0; j<n_oper; j++) {
                cu_complex result=0;
                for(int k=0; k<q_dim; k++) {
                    result += oper[j*q_dim*q_dim+idy*q_dim+k]*psi[P(n_oper, k)];
                }
                psi[P(j, idy)] = result;
            }
        }
        __syncthreads();
        
        if(idx<ntraj&&idy<n_oper) {
            double result=0;
            for(int i=0; i<q_dim; i++) {
                result += norm(psi[P(idy, i)]);
            }
            product[idx*n_oper+idy]=result;
        }
        __syncthreads();

        if(idx<ntraj&&idy<n_oper) {
            double result = 0;
            for(int i=0; i<=idy; i++) {
                result += product[idx*n_oper+i];
            }
            cumulate_weight[idx*n_oper+idy] = result;
        }
        __syncthreads();

        if(idx<ntraj&&idy==0) {
            double random_value = curand_uniform_double(&state[idx])*cumulate_weight[idx*n_oper+n_oper-1];
            int i=0;
            while(random_value>cumulate_weight[idx*n_oper+i]) {
                i++;
            }
            flag[idx]=i;
        }
        __syncthreads();

        if(idx<ntraj&&idy<q_dim) {
            psi[P(n_oper, idy)]=psi[P(flag[idx], idy)]/sqrt(product[idx*n_oper+flag[idx]]);
        }
        __syncthreads();

        if(idx<ntraj&&idy<q_dim) {
            for(int j=0; j<n_eops; j++) {
                cu_complex result=0;
                for(int k=0; k<q_dim; k++) {
                    result += eops[j*q_dim*q_dim+idy*q_dim+k]*psi[P(n_oper, k)];
                }
                psi_eops[Q(j, idy)] = result;
            }
        }
        __syncthreads();

        if(idx<ntraj&&idy<n_eops) {
            double result=0;
            for(int i=0; i<q_dim; i++) {
                result += real(conj(psi[P(n_oper, i)])*psi_eops[Q(idy, i)]);
            }
            result /= ntraj;
            atomicAdd(&expect[idy*(t_steps+1)+time+1], result);
        }
        __syncthreads();
    }
}

void solve(complex* oper_data, complex* psi0_data, complex* eops_data,
    double* expect, int q_dim, int t_steps, int n_oper, int n_eops, int ntraj) {
        cu_complex *oper, *psi0, *psi, *psi_eops, *eops;
        curandState_t *state;
        double *expect_device, *product, *cumulate_weight;
        int *flag;
        cudaMalloc(&oper, sizeof(cu_complex)*n_oper*q_dim*q_dim);
        cudaMalloc(&psi0, sizeof(cu_complex)*q_dim);
        cudaMalloc(&psi, sizeof(cu_complex)*ntraj*(n_oper+1)*q_dim);
        cudaMalloc(&psi_eops, sizeof(cu_complex)*ntraj*n_eops*q_dim);
        cudaMalloc(&eops, sizeof(cu_complex)*n_eops*q_dim*q_dim);
        cudaMalloc(&state, sizeof(curandState_t)*ntraj);
        cudaMalloc(&expect_device, sizeof(double)*n_eops*(t_steps+1));
        cudaMalloc(&product, sizeof(double)*ntraj*n_oper);
        cudaMalloc(&cumulate_weight, sizeof(double)*ntraj*n_oper);
        cudaMalloc(&flag, sizeof(int)*ntraj);
        cudaMemcpy(oper, oper_data, sizeof(cu_complex)*n_oper*q_dim*q_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(psi0, psi0_data, sizeof(cu_complex)*q_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(eops, eops_data, sizeof(cu_complex)*n_eops*q_dim*q_dim, cudaMemcpyHostToDevice);
        cudaMemset(expect_device, 0, sizeof(double)*n_eops*(t_steps+1));
        int dimY = max(max(n_eops, n_oper), q_dim);
        int dimX = 512/dimY;
        dim3 blockSize(dimX, dimY);
        int gridSize = ntraj/dimX + 1;

        CUDAsolve<<<gridSize, blockSize>>>(oper, psi0, psi, psi_eops, eops, state, expect_device, product,
            cumulate_weight, flag, q_dim, t_steps, n_oper, n_eops, ntraj);

        cudaMemcpy(expect, expect_device, sizeof(double)*n_eops*(t_steps+1), cudaMemcpyDeviceToHost);

        for(int i=0; i<n_eops; i++) {
            double result = 0;
            for(int j=0; j<q_dim; j++) {
                for(int k=0; k<q_dim; k++) {
                    result += real(eops_data[i*q_dim*q_dim+j*q_dim+k]*conj(psi0_data[j])*psi0_data[k]);
                }
            }
            expect[i*(t_steps+1)] = result;
        }
    }
}
