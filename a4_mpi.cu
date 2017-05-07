#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>
// #include <string>
#include <cuda.h>
#include <mpi.h>
using namespace std;

// module load compiler/cuda/7.5/compilervars
// module load compiler/gcc/4.9.3/compilervars
// module load mpi/mpich/3.1.4/gcc/mpivars
// module load apps/lammps/gpu

#define thread 1024

int pid;
int k;
int start=0, end, size;
string writefile;
string readfile;

int dim = -1;
vector<int> ptr, indices, data;
vector<int> B;
vector<long long> C;

int *d_dim;
int *d_ptr, *d_indices, *d_data;
int *d_B;
long long *d_C;

void parse_input(){

	// string input in ifstream not working. Don't ask why
	// stoi not working. Also don't ask why
	ifstream fin(readfile.c_str());
	string r,c,d,temp;
	fin >> temp >> temp >> temp >> dim >> temp;
	fin >> r >> c >> d;
	B = vector<int>(dim);
	int current_row = 0;
	int local_ptr = 0;
	ptr.push_back(0);

	while(r[0] != 'B'){
		// cout << r << " " << c << " " << d << endl;
		int this_row = atoi(r.c_str());
		for(int i=current_row; i<this_row; ++i) ptr.push_back(local_ptr);
		indices.push_back(atoi(c.c_str()));
		data.push_back(atoi(d.c_str()));
		current_row = this_row;
		++local_ptr;
		fin >> r >> c >> d;
	}
	ptr.push_back(local_ptr);
	B[0] = atoi(c.c_str());
	B[1] = atoi(d.c_str());
	for(int i=2; i<dim; ++i) fin >> B[i];

	// auto doesn't work for some reason...
	// cout << "dim " << dim << endl;
	// cout << "ptr "; for(int i=0; i<ptr.size(); ++i) cout << ptr[i] << " "; cout << endl;
	// cout << "indices "; for(int i=0; i<indices.size(); ++i) cout << indices[i] << " "; cout << endl;
	// cout << "data "; for(int i=0; i<data.size(); ++i) cout << data[i] << " "; cout << endl;
	// cout << "B "; for(int i=0; i<B.size(); ++i) cout << B[i] << " "; cout << endl;
	// cout << "End of parsing\n";
}

void init(){
	C = vector<long long>(end-start);

	cudaMalloc((void **)&d_dim, sizeof(int));
	cudaMalloc((void **)&d_ptr, (end-start+1) * sizeof(int));
	cudaMalloc((void **)&d_indices, indices.size() * sizeof(int));
	cudaMalloc((void **)&d_data, data.size() * sizeof(int));
	cudaMalloc((void **)&d_B, B.size() * sizeof(int));
	cudaMalloc((void **)&d_C, C.size() * sizeof(long long));

	cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ptr, &ptr[start], (end-start+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, &indices[0], indices.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data, &data[0], data.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, &B[0], B.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, &C[0], C.size() * sizeof(long long), cudaMemcpyHostToDevice);
}

void anti_init(){
	cudaMemcpy(&C[0], d_C, C.size() * sizeof(long long), cudaMemcpyDeviceToHost);

	cudaFree(d_dim);
	cudaFree(d_ptr);
	cudaFree(d_indices);
	cudaFree(d_data);
	cudaFree(d_B);
}

__global__ void kernel(
		int *dim, 
		int *ptr, 
		int *indices, 
		int *data,
		int *B,
		long long *C)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	if(row < *dim){
		long long sum = 0;
		for(int i=ptr[row]; i<ptr[row+1]; ++i)
			sum += data[i] * B[indices[i]];
		C[row] = sum;
		// printf("Final row %d sum %d\n", row, sum);
	}
}

__global__ void kernel_complex(
		int *dim, 
		int *ptr, 
		int *indices, 
		int *data,
		int *B,
		long long *C)
{
	__shared__ long long sum[thread];
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int row = id/32;
	int lane = id & (32 - 1);
	sum[tid] = 0;
	if(row < *dim){
		// printf("row %d sum %d tid %d lane %d\n", row, sum[tid], tid, lane);
		for(int i=ptr[row] + lane; i<ptr[row+1]; i+=32){
			sum[tid] += (long long)data[i] * (long long)B[indices[i]];
		}
		// printf("row %d sum %d tid %d lane %d\n", row, sum[tid], tid, lane);
	}
	// __syncthreads();
	if(lane < 16) sum[tid] += sum[tid + 16];
	if(lane < 8) sum[tid] += sum[tid + 8];
	if(lane < 4) sum[tid] += sum[tid + 4];
	if(lane < 2) sum[tid] += sum[tid + 2];
	if(lane < 1) sum[tid] += sum[tid + 1];
	if(lane == 0) C[row] = sum[tid];
	// printf("Final row %d sum %d tid %d \n", row, sum[tid], tid);
}

void send_receive_data(){
	if(pid == 0){
		start = 0;
		end = dim/k;
		size = ptr[end] - ptr[start];
		for(int i=1; i<k; ++i){
			// Send indices and data to process i
			int start = i*dim/k;
			int end = (i+1)*dim/k;
			int size = ptr[end] - ptr[start];
			MPI_Send(&indices[ptr[start]], size, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&data[ptr[start]], size, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
		indices.resize(ptr[dim/k]);
		data.resize(ptr[dim/k]);
	}
	else{
		start = pid*dim/k;
		end = (pid+1)*dim/k;
		size = ptr[end] - ptr[start];
		indices = vector<int>(size);
		data = vector<int>(size);
		// printf("pid %d start %d end %d size %d ptr_st %d ptr_end %d\n", pid, start, end, size, ptr[start], ptr[end]);
		MPI_Recv(&indices[0], size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&data[0], size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for(int i=end; i>=start; --i) ptr[i] -= ptr[start];
	}
}

void write(){
	// Write to file
	for(int j = 0; j < k; ++j) {
		ofstream fout;
		fout.open(writefile.c_str(), std::ios::app);
		if(pid == j) {
			for(int i=0; i<C.size(); ++i) {
				fout << C[i] << "\n";
			}
		}
		fout.close();
		MPI_Barrier(MPI_COMM_WORLD);
	}

	// int next = 1;
	// ofstream fout;
	// if(pid != 0) MPI_Recv(&next, 1, MPI_INT, pid-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// if(pid != 0) fout.open(writefile.c_str(), std::fstream::app);
	// else fout.open(writefile.c_str());
	// for(int i=0; i<C.size(); ++i) fout << C[i] << "\n";
	// fout.close();
	// if(pid != k-1) MPI_Send(&next, 1, MPI_INT, pid+1, 0, MPI_COMM_WORLD);
}

int main(int argc, char const *argv[])
{
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &k);

	readfile = string(argv[1]);
	writefile = string(argv[2]);
	if(pid == 0) parse_input();
	MPI_Bcast(&dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(pid != 0) ptr = vector<int>(dim+1);
	if(pid != 0) B = vector<int>(dim);
	MPI_Bcast(&ptr[0], dim+1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&B[0], dim, MPI_INT, 0, MPI_COMM_WORLD);
	send_receive_data();

	// if(pid == 0){
	// 	cout << "pid " << pid << " size " << size << endl;
	// 	cout << "pid " << pid << " dim " << dim << endl;
	// 	cout << "pid " << pid << " ptr "; for(int i=start; i<=end; ++i) cout << ptr[i] << " "; cout << endl;
	// 	cout << "pid " << pid << " indices "; for(int i=0; i<indices.size(); ++i) cout << indices[i] << " "; cout << endl;
	// 	cout << "pid " << pid << " data "; for(int i=0; i<data.size(); ++i) cout << data[i] << " "; cout << endl;
	// 	cout << "pid " << pid << " B "; for(int i=0; i<B.size(); ++i) cout << B[i] << " "; cout << endl;
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	// if(pid == 1){
	// 	cout << "pid " << pid << " size " << size << endl;
	// 	cout << "pid " << pid << " dim " << dim << endl;
	// 	cout << "pid " << pid << " ptr "; for(int i=start; i<=end; ++i) cout << ptr[i] << " "; cout << endl;
	// 	cout << "pid " << pid << " indices "; for(int i=0; i<indices.size(); ++i) cout << indices[i] << " "; cout << endl;
	// 	cout << "pid " << pid << " data "; for(int i=0; i<data.size(); ++i) cout << data[i] << " "; cout << endl;
	// 	cout << "pid " << pid << " B "; for(int i=0; i<B.size(); ++i) cout << B[i] << " "; cout << endl;
	// }

	init();

	int block = ceil(1.0f * C.size()/(thread/32));
	kernel_complex<<<block, thread>>>(d_dim, d_ptr, d_indices, d_data, d_B, d_C);
	// int block = ceil(1.0f * C.size()/thread);
	// kernel<<<block, thread>>>(d_dim, d_ptr, d_indices, d_data, d_B, d_C);
	cout << "block " << block << " C " << C.size() << endl;

	anti_init();
	
	// cout << "C for pid " << pid << " : "; for(int i=0; i<C.size(); ++i) cout << C[i] << " "; cout << endl;
	write();
	MPI_Finalize();
	return 0;
}