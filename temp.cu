#include <iostream>
#include <vector>
using namespace std;

const int N = 512;

__global__ void add(int *a, int *b, int *c){
	int tid = threadIdx.x;
	c[tid] = a[tid] + b[tid];
}

int main(int argc, char const *argv[])
{
	// int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// a = (int *)malloc(size);
	// b = (int *)malloc(size);
	// c = (int *)malloc(size);

	vector<int> a(N), b(N), c(N);

	for(int i=0; i<N; ++i){
		a[i] = i;
		b[i] = i;
	}

	cudaMemcpy(d_a, &a[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b[0], size, cudaMemcpyHostToDevice);

	add<<<1,N>>>(d_a, d_b, d_c);

	cudaMemcpy(&c[0], d_c, size, cudaMemcpyDeviceToHost);

	for(int i=0; i<N; ++i){
		cout << c[i] << " ";
	}

	// free(a);
	// free(b);
	// free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}