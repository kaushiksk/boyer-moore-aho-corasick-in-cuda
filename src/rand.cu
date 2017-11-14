#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <iostream>
using namespace std;





__global__ void boyer_moore (int *g){

	char s_shared[32768];
	for(long j=0;j<10000000;j++){
	for(int i=0;i<32;i++){
		//s_shared[i*1024+(threadIdx.x*4)+(threadIdx.x%4)] = char(threadIdx.x%256);
		s_shared[i*1024+(threadIdx.x)] = char(threadIdx.x%256);
		if(threadIdx.x%32==0){
			float randum = 0;
			randum = g[threadIdx.x];
			randum/=45.0;
		}
		float a = g[threadIdx.x];
		a++;
		a = a/(a+threadIdx.x);
	}
	}
	if(threadIdx.x==0);
		//printf("%d\n",blockIdx.x);
}



int8_t h_string[1000000];
int8_t h_pat[100];

int main(int argc, char const *argv[]) {
	cudaEvent_t start,stop;
	int g[1024];
	for(int i=0;i<1024;i++)
		g[i] = i;
	int *d_g;
	cudaMalloc(&d_g,1024*sizeof(int));
	cudaMemcpy(&d_g,g,sizeof(int)*1024,cudaMemcpyHostToDevice);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    boyer_moore<<<1000,1024>>>(d_g);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millsec=0;
    cudaEventElapsedTime(&millsec,start,stop);
    cout<<"This is time elapsed "<<millsec;
    return 0;
}
