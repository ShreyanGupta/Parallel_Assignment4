#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
// #include <string>
#include <cuda.h>
using namespace std;

// module load compiler/cuda/7.5/compilervars
// module load compiler/gcc/4.9.3/compilervars
// module load mpi/mpich/3.1.4/gcc/mpivars
// module load apps/lammps/gpu

int dim = -1;
vector<int> ptr, indices, data;
vector<int> B,C;

int *d_dim;
int *d_ptr, *d_indices, *d_data;
int *d_B, *d_C;

void parse_input(string file){

	// string input in ifstream not working. Don't ask why
	// stoi not working. Also don't ask why
	ifstream fin(file.c_str());
	string r,c,d,temp;
	fin >> temp >> temp >> temp >> dim >> temp;
	fin >> r >> c >> d;
	B = vector<int>(dim);
	C = vector<int>(dim);
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
	for(int i=0; i<ptr.size(); ++i) 	cout << ptr[i] << " "; cout << endl;
	for(int i=0; i<indices.size(); ++i) cout << indices[i] << " "; cout << endl;
	for(int i=0; i<data.size(); ++i) 	cout << data[i] << " "; cout << endl;
	for(int i=0; i<B.size(); ++i) 		cout << B[i] << " "; cout << endl;
}

void init(){
	cudaMalloc((void **)&d_dim, sizeof(int));
	cudaMalloc((void **)&d_ptr, ptr.size() * sizeof(int));
	cudaMalloc((void **)&d_indices, indices.size() * sizeof(int));
	cudaMalloc((void **)&d_data, data.size() * sizeof(int));
	cudaMalloc((void **)&d_B, B.size() * sizeof(int));
	cudaMalloc((void **)&d_C, C.size() * sizeof(int));

	cudaMemcpy(d_dim, &dim, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ptr, &ptr[0], ptr.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, &indices[0], indices.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data, &data[0], data.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, &B[0], B.size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, &C[0], C.size() * sizeof(int), cudaMemcpyHostToDevice);
}

void anti_init(){
	cudaMemcpy(&C[0], d_C, C.size() * sizeof(int), cudaMemcpyDeviceToHost);

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
		int *C)
{
	int tid = threadIdx.x;
	C[tid] = B[tid];
}

int main(int argc, char const *argv[])
{
	string file = "input1.txt";
	parse_input(file);
	init();

	kernel<<<1,dim>>>(d_dim, d_ptr, d_indices, d_data, d_B, d_C);

	anti_init();
	for(int i=0; i<C.size(); ++i) cout << C[i] << " "; cout << endl;
	return 0;
}