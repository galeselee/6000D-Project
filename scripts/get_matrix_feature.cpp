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
#include <iostream>


// only linux
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <linux/version.h>

void create_csv_file(std::ofstream &p, std::string out_csv) {
  // p.open(out_csv.c_str(),std::ios::out|std::ios::app);
  p.open(out_csv.c_str(),std::ios::out|std::ios::trunc);
  p << "filename" << "," 
    << "nrow" << ","
    << "ncol" << ","
    << "nnz" << ","
    << "sparsity" << ","
	<< "nrow_contain_nz" << ","
    << "average_dgree" << ","
    << "std_nnz_row" << std::endl;
}

int get_file_size(std::string filename) {
  FILE *fp=fopen(filename.c_str(),"r");
  if(!fp) return -1;
  fseek(fp,0L,SEEK_END);
  int size=ftell(fp);
  fclose(fp);
  return size;
}

double get_std(std::vector<int> &data, double avg) {
	double var = 0.0f;
	for(int ii = 0; ii < data.size(); ii++) {
		var += (avg - data[ii]) * (avg - data[ii]);
	}
	return std::sqrt(var/data.size()) ;
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

  for (int ii = 0; ii < filepaths.size(); ii++) {
	if (get_file_size(filepaths[ii]) < 10e9) {
		int M;                        // number of A-rows
		int K;                              // number of A-columns
		int nnz;                            // number of non-zeros in A
		std::vector<int> csr_indptr_buffer; // buffer for indptr array in CSR format
		std::vector<int>
			csr_indices_buffer; 
		std::vector<int> len_row;

		read_mtx_file_fast(filepaths[ii].c_str(), M, K, nnz, csr_indptr_buffer, csr_indices_buffer, len_row);
		//read_mtx_file_fast((root_dir+"/cage3.mtx").c_str(), M, K, nnz, csr_indptr_buffer, csr_indices_buffer, len_row);
		double avg = 0.0f;
		for (int ii_row = 0; ii_row < len_row.size(); ii_row++) {
			avg += len_row[ii_row];
		} 
		avg /= len_row.size();
		double std_row = get_std(len_row, avg);
		p << filenames[ii] << ","
		  << M << "," << K << ","
		  << nnz << "," << (double)nnz/M/K << "," << len_row.size() << ","
		  << avg << "," << std_row << std::endl;
	} else {
		continue;
	}
     std::cout << "[INFO] Start " << ii+1 << "-th" 
              << "/" << filepaths.size() << " file, filename is "
             << filenames[ii] << std::endl;
     init_ipcdevice(s_device);
     pid_t pid = fork();
     if (pid == 0) {
       // It will cost much time and fail finally.
       if (filenames[ii] == "nlpkkt240.mtx") {
         exit(EXIT_FAILURE);
       }
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
           << s_device->N_B << ",";
         for (int p_ii = 0; p_ii < 7; p_ii++)  {
           p << s_device->time[p_ii] << ",";
         }
         p << s_device->index << ","
           << alg_names[s_device->index] << std::endl;

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
