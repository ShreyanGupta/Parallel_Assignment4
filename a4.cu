#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda.h>
using namespace std;

int dim = -1;
vector<int> ptr, index, data;
vector<int> B;

void parse_input(string file){
	ifstream fin(file);
	string r,c,d,temp;
	fin >> temp >> temp >> temp >> dim >> temp;
	fin >> r >> c >> d;
	B = vector<int>(d);
	int current_row = 0
	int local_ptr = 0;
	ptr.push_back(0);
	// ptr = vector<int>(current_row+1, local_ptr);

	while(r[0] != 'B'){
		int this_row = stoi(r);
		for(int i=current_row; i<this_row; ++i) ptr.push_back(local_ptr);
		index.push_back(stoi(c));
		data.push_back(stoi(d));
		++local_ptr;
		fin >> r >> c >> d;
	}
	ptr.push_back(local_ptr);
	B[0] = stoi(c); B[1] = stoi(d);
	for(int i=2; i<dim; ++i) fin >> B[i];

	for(auto i : ptr) cout << i << " "; cout << endl;
	for(auto i : index) cout << i << " "; cout << endl;
	for(auto i : data) cout << i << " "; cout << endl;
}

int main(int argc, char const *argv[])
{
	
	return 0;
}