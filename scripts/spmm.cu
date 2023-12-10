#include "../src/kernel/gespmm.h" 
#include "../src/util/sp_util.hpp"        
#include <cstdlib>           
#include <cuda_runtime_api.h> 
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <fstream>
#include <string>
#include <dirent.h>
#include <iostream>
#include <cmath>
#include <algorithm>


// only linux
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <linux/version.h>


std::vector<std::string> alg_names{
    "CuSparse",
    "SEQREDUCE_ROWBALANCE",
    "PARREDUCE_ROWBALANCE",
    "SEQREDUCE_NNZBALANCE",
    "PARREDUCE_NNZBALANCE",
    "ROWCACHING_ROWBALANCE",
    "ROWCACHING_NNZBALANCE"
};

typedef struct ipcDevices_st
{
  int nrow;
  int ncol;
  int nnz;
  float sparsity;
  float dgree;
  float std_node_dgree;
  int N_B;
  float time[8];
  float speed[8];
  int index;
  bool success;
} ipcDevices_t;

void init_ipcdevice(ipcDevices_t *s_device) {
  s_device->nrow = 0;
  s_device->ncol = 0;
  s_device->nnz = 0;
  s_device->sparsity = 0.0;
  s_device->index = -1;
  s_device->N_B = -1;
  s_device->success = false;
}

void create_csv_file(std::ofstream &p, std::string out_csv) {
  // p.open(out_csv.c_str(),std::ios::out|std::ios::app);
  p.open(out_csv.c_str(),std::ios::out|std::ios::trunc);
  p << "filename" << "," 
    << "nrow" << ","
    << "ncol" << ","
    << "nnz" << ","
    << "sparsity" << ","
    << "degree" << ","
    << "std_node_dgree" << ","
    << "N" << ","
    << "cusparse" << "," 
    << "SEQREDUCE_ROWBALANCE" <<","
    << "SR_RB_speed" << ","
    << "PARREDUCE_ROWBALANCE" << ","
    << "PR_RB_speed" << ","
    << "SEQREDUCE_NNZBALANCE" << ","
    << "SR_NB_speed" << ","
    << "PARREDUCE_NNZBALANCE" << ","
    << "PR_NB_speed" << ","
    << "ROWCACHING_ROWBALANCE" << ","
    << "RC_RB_speed" << ","
    << "ROWCACHING_NNZBALANCE" << ","
    << "RC_NB_speed" << ","
    << "THE_BEST_ALG_INDEX" << ","
    << "THE_BEST_ALG_NAME" << ","
    << "THE_BEST_ALG_TIME" << ","
    << "THE_BEST_ALG_SPEED" << std::endl;
}

//void bench_mat(int id, std::string filepath, std::string filename, int N, ipcDevices_t* s_device) {
void bench_mat(int id, std::string filepath, std::string filename, int N, ipcDevices_t*& s_device) {
  // std::string filepath = "/home/lzy/spmm/collected_data/1138_bus.mtx";
  // std::string filename = "1138_bus.mtx";


  cudaError_t status_cuda;
  cusparseStatus_t status_cusparse;

  std::vector<float> kernel_time_vec;

  int M;                              // number of A-rows
  int K;                              // number of A-columns
  int nnz;                            // number of non-zeros in A
  std::vector<int> csr_indptr_buffer; // buffer for indptr array in CSR format
  std::vector<int>
      csr_indices_buffer; 
  std::vector<int> row_len;

  read_mtx_file_fast(filepath.c_str(), M, K, nnz, csr_indptr_buffer, csr_indices_buffer, row_len);

  float *B_h = NULL, *C_h = NULL, *csr_values_h = NULL;
  float *B_d = NULL, *C_d = NULL, *csr_values_d = NULL;
  int *csr_indptr_d = NULL, *csr_indices_d = NULL;

  B_h = (float *)malloc(sizeof(float) * K * N);
  C_h = (float *)malloc(sizeof(float) * M * N);
  csr_values_h = (float *)malloc(sizeof(float) * nnz);
  if (!B_h || !C_h || !csr_values_h) {
    std::cout << "[ERROR] Host allocation failed, filename = " 
              << filename << ", id = " << id << std::endl;
    // return EXIT_FAILURE;
  }

  fill_random(csr_values_h, nnz);
  fill_random(B_h, K * N);

  status_cuda = (cudaMalloc((void **)&B_d, sizeof(float) * K * N));
  status_cuda = (cudaMalloc((void **)&C_d, sizeof(float) * M * N));
  status_cuda = (cudaMalloc((void **)&csr_values_d, sizeof(float) * nnz));
  status_cuda = (cudaMalloc((void **)&csr_indptr_d, sizeof(int) * (M + 1)));
  status_cuda = (cudaMalloc((void **)&csr_indices_d, sizeof(int) * nnz));

  status_cuda = (
      cudaMemcpy(B_d, B_h, sizeof(float) * K * N, cudaMemcpyHostToDevice));
  status_cuda = (cudaMemset(C_d, 0x0, sizeof(float) * M * N));
  status_cuda = (cudaMemcpy(csr_values_d, csr_values_h, sizeof(float) * nnz,
                        cudaMemcpyHostToDevice));
  status_cuda = (cudaMemcpy(csr_indptr_d, csr_indptr_buffer.data(),
                        sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
  status_cuda = (cudaMemcpy(csr_indices_d, csr_indices_buffer.data(),
                        sizeof(int) * nnz, cudaMemcpyHostToDevice));

  cusparseHandle_t handle;
  cusparseSpMatDescr_t csrDescr;
  cusparseDnMatDescr_t dnMatInputDescr, dnMatOutputDescr;
  float alpha = 1.0f, beta = 0.0f;

  status_cusparse = (cusparseCreate(&handle));

  status_cusparse = (cusparseCreateCsr(
      &csrDescr, M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d,
      CUSPARSE_INDEX_32I, // index 32-integer for indptr
      CUSPARSE_INDEX_32I, // index 32-integer for indices
      CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_32F // datatype: 32-bit float real number
      ));

  status_cusparse = (cusparseCreateDnMat(&dnMatInputDescr, K, N, N, B_d, CUDA_R_32F,
                                    CUSPARSE_ORDER_ROW));
  status_cusparse = (cusparseCreateDnMat(&dnMatOutputDescr, M, N, N, C_d,
                                    CUDA_R_32F, CUSPARSE_ORDER_ROW));

  size_t workspace_size;
  status_cusparse = (cusparseSpMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrDescr, dnMatInputDescr,
      &beta, dnMatOutputDescr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
      &workspace_size));

  void *workspace = NULL;
  status_cuda = (cudaMalloc(&workspace, workspace_size));

  status_cusparse = (cusparseSpMM(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                              CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                              &alpha, csrDescr, dnMatInputDescr, &beta,
                              dnMatOutputDescr, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, workspace));
  status_cuda = (
      cudaMemcpy(C_h, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  int the_best_alg_index = 0;
  float time_best_alg = FLT_MAX;

  GpuTimer gpu_timer;
  int warmup_iter = 3;
  int repeat_iter = 10;
  for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
    if (iter == warmup_iter) {
      gpu_timer.start();
    }
    cusparseSpMM(handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE, // opA
                CUSPARSE_OPERATION_NON_TRANSPOSE, // opB
                &alpha, csrDescr, dnMatInputDescr, &beta, dnMatOutputDescr,
                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace);
  }
  gpu_timer.stop();

  float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
  time_best_alg = kernel_dur_msecs;

  kernel_time_vec.push_back(kernel_dur_msecs);

  SpMatCsrDescr_t spmatA{M, K, nnz, csr_indptr_d, csr_indices_d, csr_values_d};
  gespmmAlg_t algs[] = {
      GESPMM_ALG_SEQREDUCE_ROWBALANCE,
      GESPMM_ALG_PARREDUCE_ROWBALANCE,
      GESPMM_ALG_SEQREDUCE_NNZBALANCE,
      GESPMM_ALG_PARREDUCE_NNZBALANCE,
      GESPMM_ALG_ROWCACHING_ROWBALANCE,
      GESPMM_ALG_ROWCACHING_NNZBALANCE
  };

  // hard code 6
  for (int alg_ii = 0; alg_ii < 6; alg_ii++) {
    auto alg = algs[alg_ii];
    status_cuda = (cudaMemset(C_d, 0x0, sizeof(float) * M * N));
    gespmmCsrSpMM(spmatA, B_d, N, C_d, true, alg);
    cudaDeviceSynchronize();
    status_cuda = (
        cudaMemcpy(C_h, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    GpuTimer gpu_timer;
    int warmup_iter = 3;
    int repeat_iter = 10;
    for (int iter = 0; iter < warmup_iter + repeat_iter; iter++) {
      if (iter == warmup_iter) {
        gpu_timer.start();
      }
      gespmmCsrSpMM(spmatA, B_d, N, C_d, true, alg);
    }
    gpu_timer.stop();
    float kernel_dur_msecs = gpu_timer.elapsed_msecs() / repeat_iter;
    kernel_time_vec.push_back(kernel_dur_msecs);
    if (kernel_dur_msecs < time_best_alg) {
      the_best_alg_index = alg_ii + 1;
      time_best_alg = kernel_dur_msecs;
    }
  }

  if (B_h)
    free(B_h);
  if (C_h)
    free(C_h);
  if (csr_values_h)
    free(csr_values_h);
  if (B_d)
    status_cuda = (cudaFree(B_d));
  if (C_d)
    status_cuda = (cudaFree(C_d));
  if (csr_values_d)
    status_cuda = (cudaFree(csr_values_d));
  if (csr_indptr_d)
    status_cuda = (cudaFree(csr_indptr_d));
  if (csr_indices_d)
    status_cuda = (cudaFree(csr_indices_d));
  if (workspace)
    status_cuda = (cudaFree(workspace));

  status_cusparse = (cusparseDestroyDnMat(dnMatInputDescr));
  status_cusparse = (cusparseDestroyDnMat(dnMatOutputDescr));
  status_cusparse = (cusparseDestroySpMat(csrDescr));
  status_cusparse = (cusparseDestroy(handle));
  if (status_cuda != cudaSuccess || status_cusparse != CUSPARSE_STATUS_SUCCESS) {
    s_device->success = false;
  } else {
    s_device->nrow = M;
    s_device->ncol = K;
    s_device->nnz = nnz;
    s_device->sparsity = (float)nnz / M / K;
    s_device->dgree = (float)nnz / row_len.size();
    float var_nnz_row = 0.0f;
    for (int ii = 0; ii < row_len.size(); ii++) {
      var_nnz_row += (s_device->dgree - row_len[ii]) * (s_device->dgree - row_len[ii]);
    }
    s_device->std_node_dgree = std::sqrt(var_nnz_row/row_len.size()) ;
    s_device->N_B = N;
    s_device->index = the_best_alg_index;
    s_device->time[0] = kernel_time_vec[0];
    s_device->time[1] = kernel_time_vec[1];
    s_device->time[2] = kernel_time_vec[2];
    s_device->time[3] = kernel_time_vec[3];
    s_device->time[4] = kernel_time_vec[4];
    s_device->time[5] = kernel_time_vec[5];
    s_device->time[6] = kernel_time_vec[6];
    s_device->time[7] = kernel_time_vec[s_device->index];
    s_device->speed[0] = kernel_time_vec[0] / kernel_time_vec[0];
    s_device->speed[1] = kernel_time_vec[0] / kernel_time_vec[1];
    s_device->speed[2] = kernel_time_vec[0] / kernel_time_vec[2];
    s_device->speed[3] = kernel_time_vec[0] / kernel_time_vec[3];
    s_device->speed[4] = kernel_time_vec[0] / kernel_time_vec[4];
    s_device->speed[5] = kernel_time_vec[0] / kernel_time_vec[5];
    s_device->speed[6] = kernel_time_vec[0] / kernel_time_vec[6];
    s_device->speed[7] = kernel_time_vec[0] / kernel_time_vec[s_device->index];
    s_device->success = true;
  }
  cudaDeviceReset();
  exit(EXIT_SUCCESS);
}

int main(int argc, const char **argv) {
  if (argc < 3) {
    printf("Require command-line argument: the rootdir of "
           ".mtx format and output csv path\n");
    return EXIT_FAILURE;
  }

  std::string root_dir = argv[1];
  std::string out_csv = argv[2];
  int N = 128;
  if (argc > 3) {
    N = atoi(argv[3]);
  }
  std::ofstream p;

  create_csv_file(p, out_csv);

  std::vector<std::string> filenames;
  std::vector<std::string> filepaths;
  DIR *pDir;
  struct dirent* ptr;
  if(!(pDir = opendir(root_dir.c_str()))) {
    printf("MAT FILE ROOT DIR doesn't exist\n");
    return EXIT_FAILURE;
  }

  while((ptr = readdir(pDir))!=0) {
      if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
          filepaths.push_back(root_dir + "/" + ptr->d_name);
          filenames.push_back(ptr->d_name);
      }
  }

  ipcDevices_t *s_device = (ipcDevices_t *) mmap(NULL, sizeof(*s_device),
                                PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);

  for (int ii = 0; ii < filepaths.size(); ii++) {
    std::cout << "[INFO] Start " << ii+1 << "-th" 
              << "/" << filepaths.size() << " file, filename is "
              << filenames[ii] << std::endl;
    init_ipcdevice(s_device);
    pid_t pid = fork();
    if (pid == 0) {
      bench_mat(ii, filepaths[ii], filenames[ii], N, s_device);
    }
    else {
      int status;
      waitpid(pid, &status, 0);
      if (s_device->success == true) {
        p << filenames[ii] << ","
          << s_device->nrow << ","
          << s_device->ncol << ","
          << s_device->nnz << ","
          << s_device->sparsity << ","
          << s_device->dgree << ","
          << s_device->std_node_dgree << ","
          << s_device->N_B << ","
          << s_device->time[0] << ",";
        for (int p_ii = 1; p_ii < 7; p_ii++)  {
          p << s_device->time[p_ii] << ",";
          p << s_device->speed[p_ii] << ",";
        }
        p << s_device->index << ","
          << alg_names[s_device->index] << "," 
          << s_device->time[7] << ","
          << s_device->speed[7] << std::endl;

        std::cout << "[INFO] Successful " << ii+1 << "-th" 
          << "/" << filepaths.size() << " file, filename is "
          << filenames[ii] << std::endl; 
      } else {
        // No data if error
        std::cout << "[WARN] Failure " << ii+1 << "-th" 
          << "/" << filepaths.size() << " file, filename is "
          << filenames[ii] << std::endl; 
      }
    }
  }

  p.close();

  return EXIT_SUCCESS;
}
