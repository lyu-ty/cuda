#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <mpi.h>

int M, N;
double X1 = 0.0, X2 = 1.0;
double Y1 = -1.0, Y2 = 1.0;

double h1, h2;
double epsilon;
double TOL = 2e-6;
int MAX_ITER = 10000;

// ===================== 全局计时器（每个 MPI 进程局部累加，最后用 MPI_Reduce 汇总） =====================

// dot_local_cuda 中 GPU 相关
double g_time_dot_total    = 0.0;
double g_time_dot_h2d      = 0.0;
double g_time_dot_d2h      = 0.0;
double g_time_dot_kernel   = 0.0;

// axpy_cuda 中 GPU 相关
double g_time_axpy_total   = 0.0;
double g_time_axpy_h2d     = 0.0;
double g_time_axpy_d2h     = 0.0;
double g_time_axpy_kernel  = 0.0;

// 差分算子 chafen_mpi
double g_time_chafen_total   = 0.0;
double g_time_chafen_comm    = 0.0;
double g_time_chafen_compute = 0.0;

// 预条件子 minv_local
double g_time_minv_total   = 0.0;

// MPI_Allreduce（用在 dot_global）
double g_time_mpi_allreduce = 0.0;

// CG 主迭代（在 gonge_mpi 里赋值，在 main 里 Reduce）
double g_time_cg_loop = 0.0;


// ===================== CUDA 辅助宏与核函数 =====================

#define CUDA_CALL(call)                                                       \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, -1);                                    \
        }                                                                     \
    } while (0)

__global__ void dot_kernel(int n, const double* a, const double* b, double* result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 简单全局原子加
        atomicAdd(result, a[idx] * b[idx]);
    }
}

__global__ void axpy_kernel(int n, double alpha, const double* x, double* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] += alpha * x[idx];
    }
}

// 在单个 MPI 进程上，用 GPU 做本地点积，并统计时间
double dot_local_cuda(const std::vector<double>& A,
                      const std::vector<double>& B) {
    int n = static_cast<int>(A.size());
    if (n == 0) return 0.0;

    double t_total_start = MPI_Wtime();

    double *dA = nullptr, *dB = nullptr, *dResult = nullptr;
    CUDA_CALL(cudaMalloc(&dA, n * sizeof(double)));
    CUDA_CALL(cudaMalloc(&dB, n * sizeof(double)));
    CUDA_CALL(cudaMalloc(&dResult, sizeof(double)));

    double t_h2d_start = MPI_Wtime();
    CUDA_CALL(cudaMemcpy(dA, A.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dB, B.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(dResult, 0, sizeof(double)));
    double t_h2d_end = MPI_Wtime();

    int block = 256;
    int grid = (n + block - 1) / block;

    double t_kernel_start = MPI_Wtime();
    dot_kernel<<<grid, block>>>(n, dA, dB, dResult);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    double t_kernel_end = MPI_Wtime();

    double local = 0.0;
    double t_d2h_start = MPI_Wtime();
    CUDA_CALL(cudaMemcpy(&local, dResult, sizeof(double), cudaMemcpyDeviceToHost));
    double t_d2h_end = MPI_Wtime();

    CUDA_CALL(cudaFree(dA));
    CUDA_CALL(cudaFree(dB));
    CUDA_CALL(cudaFree(dResult));

    double t_total_end = MPI_Wtime();

    g_time_dot_total  += (t_total_end - t_total_start);
    g_time_dot_h2d    += (t_h2d_end - t_h2d_start);
    g_time_dot_kernel += (t_kernel_end - t_kernel_start);
    g_time_dot_d2h    += (t_d2h_end - t_d2h_start);

    return local;
}

// 在单个 MPI 进程上，用 GPU 做 y += alpha * x，并统计时间
void axpy_cuda(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    int n = static_cast<int>(x.size());
    if (n == 0) return;

    double t_total_start = MPI_Wtime();

    double *dX = nullptr, *dY = nullptr;
    CUDA_CALL(cudaMalloc(&dX, n * sizeof(double)));
    CUDA_CALL(cudaMalloc(&dY, n * sizeof(double)));

    double t_h2d_start = MPI_Wtime();
    CUDA_CALL(cudaMemcpy(dX, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dY, y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    double t_h2d_end = MPI_Wtime();

    int block = 256;
    int grid = (n + block - 1) / block;

    double t_kernel_start = MPI_Wtime();
    axpy_kernel<<<grid, block>>>(n, alpha, dX, dY);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    double t_kernel_end = MPI_Wtime();

    double t_d2h_start = MPI_Wtime();
    CUDA_CALL(cudaMemcpy(y.data(), dY, n * sizeof(double), cudaMemcpyDeviceToHost));
    double t_d2h_end = MPI_Wtime();

    CUDA_CALL(cudaFree(dX));
    CUDA_CALL(cudaFree(dY));

    double t_total_end = MPI_Wtime();

    g_time_axpy_total  += (t_total_end - t_total_start);
    g_time_axpy_h2d    += (t_h2d_end - t_h2d_start);
    g_time_axpy_kernel += (t_kernel_end - t_kernel_start);
    g_time_axpy_d2h    += (t_d2h_end - t_d2h_start);
}

// ===================== 原有数学/网格函数 =====================

int CheckRegion(double x, double y) {
    if (x > y * y && x < 1.0) {
        return 1;
    }
    else {
        return -1;
    }
}

// 解析计算  ---PDF(11)
double shujiexi(double x, double y_start, double y_end) {
    if (x >= 1.0) return 0.0;
    if (x <= 0.0) return 0.0;

    double y_bian = std::sqrt(x);

    double y_low = std::max(y_start, -1.0);
    double y_high = std::min(y_end, 1.0);

    if (y_low >= y_high) return 0.0;

    double y_in_D_low = std::max(y_low, -y_bian);
    double y_in_D_high = std::min(y_high, y_bian);

    return std::max(y_in_D_high - y_in_D_low, 0.0);
}

double computeaij(double x, double y_bottom, double y_top) {
    double zonglength = y_top - y_bottom;
    if (zonglength <= 0.0) return 1.0 / epsilon;

    double length_in_D = shujiexi(x, y_bottom, y_top);
    double ratio = length_in_D / zonglength;

    return ratio * 1.0 + (1.0 - ratio) * (1.0 / epsilon);
}

double hengjiexi(double y, double x_start, double x_end) {
    if (std::abs(y) >= 1.0) return 0.0;

    double y2 = y * y;

    if (x_end <= y2) return 0.0;
    if (x_start >= 1.0) return 0.0;

    double x_start_in_D = std::max(x_start, y2);
    double x_end_in_D = std::min(x_end, 1.0);

    return std::max(x_end_in_D - x_start_in_D, 0.0);
}

double computebij(double y, double x_left, double x_right) {
    double zonglength = x_right - x_left;
    if (zonglength <= 0.0) return 1.0 / epsilon;

    double length_in_D = hengjiexi(y, x_left, x_right);
    double ratio = length_in_D / zonglength;

    return ratio * 1.0 + (1.0 - ratio) * (1.0 / epsilon);
}

double computeinD(double x_left, double x_right,
                  double y_bottom, double y_top) {

    if (x_left >= 1.0 || x_right <= 0.0) return 0.0;
    if (y_bottom >= 1.0 || y_top <= -1.0) return 0.0;

    double x_low = std::max(x_left, 0.0);
    double x_high = std::min(x_right, 1.0);
    double y_low = std::max(y_bottom, -1.0);
    double y_high = std::min(y_top, 1.0);

    if (x_low >= x_high || y_low >= y_high) return 0.0;

    double area = 0.0;
    const int segments = 100;
    double dx = (x_high - x_low) / segments;

    for (int i = 0; i < segments; i++) {
        double x = x_low + (i + 0.5) * dx;
        double y_boundary = std::sqrt(x);

        double ydi = std::max(y_low, -y_boundary);
        double ygao = std::min(y_high, y_boundary);

        if (ydi < ygao) {
            area += (ygao - ydi) * dx;
        }
    }

    return area;
}

double coefficient_Fij_analytical(double x_left, double x_right,
                                  double y_bottom, double y_top) {

    double total_area = (x_right - x_left) * (y_top - y_bottom);
    double area_in_D = computeinD(x_left, x_right, y_bottom, y_top);

    if (total_area < 1e-12) return 0.0;

    return (area_in_D / total_area) * 1.0;
}

void generate_grid(std::vector<double>& x_nodes, std::vector<double>& y_nodes) {
    h1 = (X2 - X1) / M;
    h2 = (Y2 - Y1) / N;

    x_nodes.resize(M + 1);
    y_nodes.resize(N + 1);

    for (int i = 0; i <= M; i++) {
        x_nodes[i] = X1 + i * h1;
    }
    for (int j = 0; j <= N; j++) {
        y_nodes[j] = Y1 + j * h2;
    }
}

void build(int M, int N,
           const std::vector<double>& x_nodes,
           const std::vector<double>& y_nodes,
           std::vector<double>& a_ij,
           std::vector<double>& b_ij,
           std::vector<double>& F) {

    int inner_M = M - 1;
    int inner_N = N - 1;
    F.resize(inner_M * inner_N);

    a_ij.resize((M + 1) * N);
    for (int i = 0; i <= M; i++) {
        for (int j = 0; j < N; j++) {
            double x_half = x_nodes[i];
            double y_bottom = y_nodes[j];
            double y_top = y_nodes[j + 1];

            a_ij[i * N + j] = computeaij(x_half, y_bottom, y_top);
        }
    }

    b_ij.resize(M * (N + 1));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j <= N; j++) {
            double y_half = y_nodes[j];
            double x_left = x_nodes[i];
            double x_right = x_nodes[i + 1];

            b_ij[i * (N + 1) + j] = computebij(y_half, x_left, x_right);
        }
    }

    for (int i = 0; i < inner_M; i++) {
        for (int j = 0; j < inner_N; j++) {
            double x_left = x_nodes[i];
            double x_right = x_nodes[i + 1];
            double y_bottom = y_nodes[j];
            double y_top = y_nodes[j + 1];

            F[i * inner_N + j] = coefficient_Fij_analytical(
                x_left, x_right, y_bottom, y_top);
        }
    }
}

// 选择 P1, P2, 使 P1*P2 = P，并满足长宽比要求
void choose_2d_decomposition(int P, int M, int N, int& P1, int& P2) {
    int lim = M - 1;
    int lin = N - 1;

    double bestScore = 1e100;
    int bestP1 = 1, bestP2 = P;

    for (int d = 1; d * d <= P; ++d) {
        if (P % d != 0) continue;
        int q = P / d;

        int cand[2][2] = { {d, q}, {q, d} };
        for (int c = 0; c < 2; ++c) {
            int p1 = cand[c][0];
            int p2 = cand[c][1];

            int base_m = lim / p1;
            int rem_m = lim % p1;
            int base_n = lin / p2;
            int rem_n = lin % p2;

            if (base_m == 0 || base_n == 0) continue;

            int m_vals[2] = { base_m, base_m + 1 };
            int n_vals[2] = { base_n, base_n + 1 };

            double min_ratio = 1e100, max_ratio = 0.0;

            for (int mi = 0; mi < 2; ++mi) {
                int m_loc = m_vals[mi];
                if (mi == 1 && rem_m == 0) continue;

                for (int nj = 0; nj < 2; ++nj) {
                    int n_loc = n_vals[nj];
                    if (nj == 1 && rem_n == 0) continue;

                    double ratio = double(m_loc + 1) / double(n_loc + 1);
                    min_ratio = std::min(min_ratio, ratio);
                    max_ratio = std::max(max_ratio, ratio);
                }
            }

            if (min_ratio < 0.5 || max_ratio > 2.0) continue;

            double score = std::fabs(double(lim) / p1 - double(lin) / p2);
            if (score < bestScore) {
                bestScore = score;
                bestP1 = p1;
                bestP2 = p2;
            }
        }
    }

    P1 = bestP1;
    P2 = bestP2;
}

// 计算本地块
void compute_local_block(int M, int N,
                         int P1, int P2,
                         int rank,
                         int& inner_M_local, int& inner_N_local,
                         int& i0, int& j0,
                         int& px, int& py) {
    int inner_M = M - 1;
    int inner_N = N - 1;

    px = rank / P2;
    py = rank % P2;

    int base_m = inner_M / P1;
    int rem_m = inner_M % P1;
    int base_n = inner_N / P2;
    int rem_n = inner_N % P2;

    inner_M_local = base_m + (px < rem_m ? 1 : 0);
    inner_N_local = base_n + (py < rem_n ? 1 : 0);

    i0 = px * base_m + std::min(px, rem_m);
    j0 = py * base_n + std::min(py, rem_n);
}

// ===================== MPI + CUDA 点积 / AXPY =====================

// dot_global：利用 GPU 做本地点积，然后 MPI_Allreduce，全局归约并计时 Allreduce
double dot_global(const std::vector<double>& A,
                  const std::vector<double>& B,
                  MPI_Comm comm) {
    const int n = static_cast<int>(A.size());
    if (n == 0) return 0.0;

    double local_acc = dot_local_cuda(A, B);

    double global_acc = 0.0;
    double t_allreduce_start = MPI_Wtime();
    MPI_Allreduce(&local_acc, &global_acc, 1, MPI_DOUBLE, MPI_SUM, comm);
    double t_allreduce_end = MPI_Wtime();

    g_time_mpi_allreduce += (t_allreduce_end - t_allreduce_start);

    return global_acc;
}

void axpy_2d(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    axpy_cuda(alpha, x, y);
}

// ===================== 预条件子、本地算子、CG 等 =====================

// 预条件子：对角近似（局部块）
void minv_local(int inner_M_local, int inner_N_local,
                int i0, int j0,
                const std::vector<double>& r_local,
                const std::vector<double>& a_ij,
                const std::vector<double>& b_ij,
                std::vector<double>& z_local) {
    double t_start = MPI_Wtime();

    const int stride = inner_N_local;
    int inner_M_global = M - 1;
    int inner_N_global = N - 1;

    for (int li = 0; li < inner_M_local; ++li) {
        int i = i0 + li;
        for (int lj = 0; lj < inner_N_local; ++lj) {
            int j = j0 + lj;
            int idx = li * stride + lj;

            double a_right = a_ij[(i + 1) * N + j];
            double a_left  = a_ij[i * N + j];
            double b_top   = b_ij[i * (N + 1) + (j + 1)];
            double b_bottom= b_ij[i * (N + 1) + j];

            double diag = (a_left + a_right) / (h1 * h1) +
                          (b_bottom + b_top) / (h2 * h2);

            z_local[idx] = r_local[idx] / (diag + 1e-10);
        }
    }

    double t_end = MPI_Wtime();
    g_time_minv_total += (t_end - t_start);
}

// 差分算子 Ap = A p 的 MPI 版本（带 halo 通信）—— 仍在 CPU 上
void chafen_mpi(int inner_M_global, int inner_N_global,
                int inner_M_local, int inner_N_local,
                int i0, int j0,
                const std::vector<double>& p_local,
                const std::vector<double>& a_ij,
                const std::vector<double>& b_ij,
                std::vector<double>& Ap_local,
                int P1, int P2,
                MPI_Comm comm) {

    double t_total_start = MPI_Wtime();

    int rank;
    MPI_Comm_rank(comm, &rank);
    int px = rank / P2;
    int py = rank % P2;

    int rank_im1 = (px > 0) ? ((px - 1) * P2 + py) : MPI_PROC_NULL;
    int rank_ip1 = (px < P1 - 1) ? ((px + 1) * P2 + py) : MPI_PROC_NULL;
    int rank_jm1 = (py > 0) ? (px * P2 + (py - 1)) : MPI_PROC_NULL;
    int rank_jp1 = (py < P2 - 1) ? (px * P2 + (py + 1)) : MPI_PROC_NULL;

    std::vector<double> send_im1, send_ip1, recv_im1, recv_ip1;
    std::vector<double> send_jm1, send_jp1, recv_jm1, recv_jp1;

    if (rank_im1 != MPI_PROC_NULL) {
        send_im1.resize(inner_N_local);
        recv_im1.resize(inner_N_local);
        for (int j = 0; j < inner_N_local; ++j) {
            send_im1[j] = p_local[0 * inner_N_local + j];
        }
    }
    if (rank_ip1 != MPI_PROC_NULL) {
        send_ip1.resize(inner_N_local);
        recv_ip1.resize(inner_N_local);
        for (int j = 0; j < inner_N_local; ++j) {
            send_ip1[j] = p_local[(inner_M_local - 1) * inner_N_local + j];
        }
    }
    if (rank_jm1 != MPI_PROC_NULL) {
        send_jm1.resize(inner_M_local);
        recv_jm1.resize(inner_M_local);
        for (int i = 0; i < inner_M_local; ++i) {
            send_jm1[i] = p_local[i * inner_N_local + 0];
        }
    }
    if (rank_jp1 != MPI_PROC_NULL) {
        send_jp1.resize(inner_M_local);
        recv_jp1.resize(inner_M_local);
        for (int i = 0; i < inner_M_local; ++i) {
            send_jp1[i] = p_local[i * inner_N_local + (inner_N_local - 1)];
        }
    }

    MPI_Request reqs[8];
    int rcount = 0;

    if (rank_im1 != MPI_PROC_NULL) {
        MPI_Irecv(recv_im1.data(), inner_N_local, MPI_DOUBLE, rank_im1, 0, comm, &reqs[rcount++]);
    }
    if (rank_ip1 != MPI_PROC_NULL) {
        MPI_Irecv(recv_ip1.data(), inner_N_local, MPI_DOUBLE, rank_ip1, 1, comm, &reqs[rcount++]);
    }
    if (rank_jm1 != MPI_PROC_NULL) {
        MPI_Irecv(recv_jm1.data(), inner_M_local, MPI_DOUBLE, rank_jm1, 2, comm, &reqs[rcount++]);
    }
    if (rank_jp1 != MPI_PROC_NULL) {
        MPI_Irecv(recv_jp1.data(), inner_M_local, MPI_DOUBLE, rank_jp1, 3, comm, &reqs[rcount++]);
    }

    if (rank_im1 != MPI_PROC_NULL) {
        MPI_Isend(send_im1.data(), inner_N_local, MPI_DOUBLE, rank_im1, 1, comm, &reqs[rcount++]);
    }
    if (rank_ip1 != MPI_PROC_NULL) {
        MPI_Isend(send_ip1.data(), inner_N_local, MPI_DOUBLE, rank_ip1, 0, comm, &reqs[rcount++]);
    }
    if (rank_jm1 != MPI_PROC_NULL) {
        MPI_Isend(send_jm1.data(), inner_M_local, MPI_DOUBLE, rank_jm1, 3, comm, &reqs[rcount++]);
    }
    if (rank_jp1 != MPI_PROC_NULL) {
        MPI_Isend(send_jp1.data(), inner_M_local, MPI_DOUBLE, rank_jp1, 2, comm, &reqs[rcount++]);
    }

    double t_comm = 0.0;
    if (rcount > 0) {
        double t_comm_start = MPI_Wtime();
        MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);
        double t_comm_end = MPI_Wtime();
        t_comm = (t_comm_end - t_comm_start);
    }

    Ap_local.assign(inner_M_local * inner_N_local, 0.0);
    const int stride = inner_N_local;

    double t_comp_start = MPI_Wtime();

    for (int li = 0; li < inner_M_local; ++li) {
        int i = i0 + li;
        for (int lj = 0; lj < inner_N_local; ++lj) {
            int j = j0 + lj;
            int idx = li * stride + lj;

            double p_ij = p_local[idx];

            double p_left, p_right, p_bottom, p_top;

            if (i > 0) {
                if (li > 0) {
                    p_left = p_local[(li - 1) * stride + lj];
                }
                else if (rank_im1 != MPI_PROC_NULL) {
                    p_left = recv_im1[lj];
                }
                else {
                    p_left = 0.0;
                }
            }
            else {
                p_left = 0.0;
            }

            if (i < inner_M_global - 1) {
                if (li < inner_M_local - 1) {
                    p_right = p_local[(li + 1) * stride + lj];
                }
                else if (rank_ip1 != MPI_PROC_NULL) {
                    p_right = recv_ip1[lj];
                }
                else {
                    p_right = 0.0;
                }
            }
            else {
                p_right = 0.0;
            }

            if (j > 0) {
                if (lj > 0) {
                    p_bottom = p_local[li * stride + (lj - 1)];
                }
                else if (rank_jm1 != MPI_PROC_NULL) {
                    p_bottom = recv_jm1[li];
                }
                else {
                    p_bottom = 0.0;
                }
            }
            else {
                p_bottom = 0.0;
            }

            if (j < inner_N_global - 1) {
                if (lj < inner_N_local - 1) {
                    p_top = p_local[li * stride + (lj + 1)];
                }
                else if (rank_jp1 != MPI_PROC_NULL) {
                    p_top = recv_jp1[li];
                }
                else {
                    p_top = 0.0;
                }
            }
            else {
                p_top = 0.0;
            }

            double a_right = a_ij[(i + 1) * N + j];
            double a_left  = a_ij[i * N + j];
            double b_top   = b_ij[i * (N + 1) + (j + 1)];
            double b_bottom= b_ij[i * (N + 1) + j];

            double flux_x_right = a_right * (p_right - p_ij) / h1;
            double flux_x_left  = a_left  * (p_ij - p_left)  / h1;
            double term_x = (flux_x_right - flux_x_left) / h1;

            double flux_y_top    = b_top    * (p_top - p_ij)    / h2;
            double flux_y_bottom = b_bottom * (p_ij - p_bottom) / h2;
            double term_y = (flux_y_top - flux_y_bottom) / h2;

            Ap_local[idx] = -term_x - term_y;
        }
    }

    double t_total_end = MPI_Wtime();

    g_time_chafen_total   += (t_total_end - t_total_start);
    g_time_chafen_comm    += t_comm;
    g_time_chafen_compute += (t_total_end - t_comp_start);
}

// 共轭梯度法（MPI + CUDA）
void gonge_mpi(const std::vector<double>& F_global,
               const std::vector<double>& a_ij,
               const std::vector<double>& b_ij,
               std::vector<double>& x_local,
               int P1, int P2,
               MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int inner_M_global = M - 1;
    int inner_N_global = N - 1;

    int inner_M_local, inner_N_local, i0, j0, px, py;
    compute_local_block(M, N, P1, P2, rank,
                        inner_M_local, inner_N_local,
                        i0, j0, px, py);

    int local_size = inner_M_local * inner_N_local;

    x_local.assign(local_size, 0.0);
    std::vector<double> r_local(local_size), p_local(local_size),
                        z_local(local_size), Ap_local(local_size);

    for (int li = 0; li < inner_M_local; ++li) {
        int i = i0 + li;
        for (int lj = 0; lj < inner_N_local; ++lj) {
            int j = j0 + lj;
            int local_idx = li * inner_N_local + lj;
            int global_idx = i * inner_N_global + j;
            r_local[local_idx] = F_global[global_idx];
        }
    }

    double rho = 0.0, rho_prev = 0.0, alpha = 0.0, beta = 0.0;

    if (rank == 0) {
        std::cout << "Iter\tResidual\n------\t--------\n";
    }

    double start_time = MPI_Wtime();

    double initial_residual_sq = dot_global(r_local, r_local, comm);
    double initial_residual = std::sqrt(initial_residual_sq);
    double residual_norm = initial_residual;

    int k = 0;

    double t_cg_start = MPI_Wtime();

    while (residual_norm > TOL && k < MAX_ITER) {
        minv_local(inner_M_local, inner_N_local,
                   i0, j0, r_local, a_ij, b_ij, z_local);

        rho = dot_global(r_local, z_local, comm);

        if (k == 0) {
            p_local = z_local;
        }
        else {
            beta = rho / rho_prev;
            for (int i = 0; i < local_size; ++i) {
                p_local[i] = z_local[i] + beta * p_local[i];
            }
        }

        chafen_mpi(inner_M_global, inner_N_global,
                   inner_M_local, inner_N_local,
                   i0, j0,
                   p_local, a_ij, b_ij, Ap_local,
                   P1, P2, comm);

        double pAp = dot_global(p_local, Ap_local, comm);
        alpha = rho / pAp;

        axpy_2d(alpha, p_local, x_local);
        axpy_2d(-alpha, Ap_local, r_local);

        rho_prev = rho;
        double rr = dot_global(r_local, r_local, comm);
        residual_norm = std::sqrt(rr);
        ++k;

        if (rank == 0 && (k % 100 == 0 || residual_norm < 10 * TOL)) {
            std::cout << k << "\t" << std::scientific << residual_norm << "\n";
        }
    }

    double t_cg_end = MPI_Wtime();
    g_time_cg_loop = t_cg_end - t_cg_start;

    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "------\t--------\n";
        std::cout << "Converged in " << k << " iterations\n";
        std::cout << "Final residual: " << std::scientific << residual_norm << "\n";
        std::cout << "Initial residual: " << std::scientific << initial_residual << "\n";
        std::cout << "Reduction factor: " << residual_norm / initial_residual << "\n";
        std::cout << "Solver wall-time (inner CG section): " << std::fixed << (end_time - start_time) << " seconds\n";
    }
}

// ===================== 主函数（MPI + CUDA，带时间统计输出） =====================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 程序总时间起点（MPI 已经初始化）
    double t_program_start = MPI_Wtime();

    // 初始化 CUDA 设备：简单按 rank 映射
    int devCount = 0;
    CUDA_CALL(cudaGetDeviceCount(&devCount));
    CUDA_CALL(cudaSetDevice(rank % devCount));

    // 新的命令行： ./a.out M N  （不再输入 P1 P2）
    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " M N\n";
            std::cerr << "Example: mpiexec -n 4 " << argv[0]
                      << " 40 40\n";
        }
        MPI_Finalize();
        return 1;
    }

    M = std::stoi(argv[1]);
    N = std::stoi(argv[2]);

    // 自动选择 P1, P2，使得 P1 * P2 = size
    int P1 = 1, P2 = size;
    choose_2d_decomposition(size, M, N, P1, P2);

    h1 = (X2 - X1) / M;
    h2 = (Y2 - Y1) / N;
    epsilon = std::max(h1, h2) * std::max(h1, h2);

    if (rank == 0) {
        std::cout << "=== MPI + CUDA PDE Solver ===\n";
        std::cout << "Global grid: M=" << M << ", N=" << N << "\n";
        std::cout << "MPI processes: " << size
                  << " (P1=" << P1 << ", P2=" << P2 << ")\n";
        std::cout << "epsilon = " << epsilon << "\n";
    }

    int inner_M = M - 1;
    int inner_N = N - 1;
    int base_m = inner_M / P1;
    int rem_m = inner_M % P1;
    int base_n = inner_N / P2;
    int rem_n = inner_N % P2;

    double min_ratio = 1e100, max_ratio = 0.0;
    for (int px = 0; px < P1; ++px) {
        int m_loc = base_m + (px < rem_m ? 1 : 0);
        for (int py = 0; py < P2; ++py) {
            int n_loc = base_n + (py < rem_n ? 1 : 0);
            double ratio = double(m_loc + 1) / double(n_loc + 1);
            min_ratio = std::min(min_ratio, ratio);
            max_ratio = std::max(max_ratio, ratio);
        }
    }

    if (rank == 0) {
        std::cout << "Subdomain node ratio in ["
                  << min_ratio << ", " << max_ratio << "]\n";
        if (min_ratio < 0.5 || max_ratio > 2.0) {
            std::cout << "WARNING: ratio is outside [1/2, 2].\n";
        } else {
            std::cout << "Decomposition satisfies requirement 4.\n";
        }
    }

    std::vector<double> x_nodes, y_nodes;
    generate_grid(x_nodes, y_nodes);

    std::vector<double> a_ij, b_ij, F_global;
    build(M, N, x_nodes, y_nodes, a_ij, b_ij, F_global);

    // 到这里为止算“初始化 + 建矩阵”
    double t_before_solver = MPI_Wtime();

    std::vector<double> x_local;
    gonge_mpi(F_global, a_ij, b_ij, x_local, P1, P2, MPI_COMM_WORLD);

    double t_after_solver = MPI_Wtime();

    int inner_M_local, inner_N_local, i0, j0, px, py;
    compute_local_block(M, N, P1, P2, rank,
                        inner_M_local, inner_N_local,
                        i0, j0, px, py);

    double local_min = 1e300, local_max = -1e300;
    for (double v : x_local) {
        local_min = std::min(local_min, v);
        local_max = std::max(local_max, v);
    }

    double global_min, global_max;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\nGlobal solution range: [" << global_min
                  << ", " << global_max << "]\n";
    }

    // 写文件也算在“收尾阶段”
    std::ofstream ofs;
    ofs.open(("solution_rank_" + std::to_string(rank) + ".txt").c_str());
    ofs << std::scientific << std::setprecision(15);
    ofs << "# rank " << rank << "\n";

    for (int li = 0; li < inner_M_local; ++li) {
        int i = i0 + li;
        double x = X1 + h1 * (i + 1);
        for (int lj = 0; lj < inner_N_local; ++lj) {
            int j = j0 + lj;
            double y = Y1 + h2 * (j + 1);
            int idx = li * inner_N_local + lj;
            ofs << x << " " << y << " " << x_local[idx] << "\n";
        }
    }
    ofs.close();

    double t_program_end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Local solutions written to solution_rank_*.txt\n";
        std::cout << "Example run: mpiexec -n 4 ./a.out 40 40\n";
    }

    // ===================== 汇总并打印计时信息 =====================

    double local_init  = t_before_solver - t_program_start;
    double local_solve = t_after_solver - t_before_solver;
    double local_total = t_program_end - t_program_start;

    double max_init, max_solve, max_total;
    MPI_Reduce(&local_init,  &max_init,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_solve, &max_solve, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double max_cg;
    MPI_Reduce(&g_time_cg_loop, &max_cg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // GPU dot / axpy / chafen / minv / MPI_Allreduce 的时间（sum 和 max）
    auto reduce_time = [&](double local, double &sum_out, double &max_out) {
        MPI_Reduce(&local, &sum_out, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local, &max_out, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    };

    double dot_h2d_sum, dot_h2d_max;
    double dot_d2h_sum, dot_d2h_max;
    double dot_kernel_sum, dot_kernel_max;
    double dot_total_sum, dot_total_max;

    double axpy_h2d_sum, axpy_h2d_max;
    double axpy_d2h_sum, axpy_d2h_max;
    double axpy_kernel_sum, axpy_kernel_max;
    double axpy_total_sum, axpy_total_max;

    double chafen_total_sum, chafen_total_max;
    double chafen_comm_sum,  chafen_comm_max;
    double chafen_comp_sum,  chafen_comp_max;

    double minv_total_sum, minv_total_max;

    double allreduce_sum, allreduce_max;

    reduce_time(g_time_dot_h2d,    dot_h2d_sum,    dot_h2d_max);
    reduce_time(g_time_dot_d2h,    dot_d2h_sum,    dot_d2h_max);
    reduce_time(g_time_dot_kernel, dot_kernel_sum, dot_kernel_max);
    reduce_time(g_time_dot_total,  dot_total_sum,  dot_total_max);

    reduce_time(g_time_axpy_h2d,    axpy_h2d_sum,    axpy_h2d_max);
    reduce_time(g_time_axpy_d2h,    axpy_d2h_sum,    axpy_d2h_max);
    reduce_time(g_time_axpy_kernel, axpy_kernel_sum, axpy_kernel_max);
    reduce_time(g_time_axpy_total,  axpy_total_sum,  axpy_total_max);

    reduce_time(g_time_chafen_total,   chafen_total_sum,   chafen_total_max);
    reduce_time(g_time_chafen_comm,    chafen_comm_sum,    chafen_comm_max);
    reduce_time(g_time_chafen_compute, chafen_comp_sum,    chafen_comp_max);

    reduce_time(g_time_minv_total, minv_total_sum, minv_total_max);
    reduce_time(g_time_mpi_allreduce, allreduce_sum, allreduce_max);

    if (rank == 0) {
        std::cout << "\n===== Timing summary (seconds) =====\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Total wall time (MPI_Init..Finalize) [max over ranks]: " << max_total << "\n";
        std::cout << "Initialization + grid/build          [max]: " << max_init  << "\n";
        std::cout << "Solver (gonge_mpi + CG loop)         [max]: " << max_solve << "\n";
        std::cout << "  Inner CG loop only                 [max]: " << max_cg    << "\n";

        std::cout << "\n-- GPU dot product (sum over ranks / max single rank) --\n";
        std::cout << "H2D copies : sum=" << dot_h2d_sum    << ", max=" << dot_h2d_max    << "\n";
        std::cout << "Kernel     : sum=" << dot_kernel_sum << ", max=" << dot_kernel_max << "\n";
        std::cout << "D2H copies : sum=" << dot_d2h_sum    << ", max=" << dot_d2h_max    << "\n";
        std::cout << "Total      : sum=" << dot_total_sum  << ", max=" << dot_total_max  << "\n";

        std::cout << "\n-- GPU axpy (y += alpha * x) --\n";
        std::cout << "H2D copies : sum=" << axpy_h2d_sum    << ", max=" << axpy_h2d_max    << "\n";
        std::cout << "Kernel     : sum=" << axpy_kernel_sum << ", max=" << axpy_kernel_max << "\n";
        std::cout << "D2H copies : sum=" << axpy_d2h_sum    << ", max=" << axpy_d2h_max    << "\n";
        std::cout << "Total      : sum=" << axpy_total_sum  << ", max=" << axpy_total_max  << "\n";

        std::cout << "\n-- Chafen (discrete operator + halo MPI) --\n";
        std::cout << "Total      : sum=" << chafen_total_sum << ", max=" << chafen_total_max << "\n";
        std::cout << "Comm (Wait): sum=" << chafen_comm_sum  << ", max=" << chafen_comm_max  << "\n";
        std::cout << "Compute    : sum=" << chafen_comp_sum  << ", max=" << chafen_comp_max  << "\n";

        std::cout << "\n-- Preconditioner minv_local --\n";
        std::cout << "Total      : sum=" << minv_total_sum << ", max=" << minv_total_max << "\n";

        std::cout << "\n-- MPI_Allreduce (dot_global) --\n";
        std::cout << "Total      : sum=" << allreduce_sum << ", max=" << allreduce_max << "\n";

        std::cout << "=====================================\n";
    }

    MPI_Finalize();
    return 0;
}
