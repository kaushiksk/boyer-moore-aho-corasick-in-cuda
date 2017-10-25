#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <iostream>
#include <cstring>

using namespace std;

#define NO_OF_CHARS 256
#define SHAREDMEMPERBLOCK 60
#define NUMTHREADSPERBLOCK 6
 //int n_blocks = n/block_size + (n%block_size==0?0:1);

__global__ void boyer_moore (char *d_string, int n, const char* __restrict__ pat, int m, 
	const int * __restrict__ delta1, const int * __restrict__ delta2, int offset, int cblockSize){
    int i;
   
    char s_string[SHAREDMEMPERBLOCK];

    int idx = threadIdx.x;

    for(i=0;i<offset;i++){
    	int sharedIndex = idx + i*blockDim.x;
    	int globalIndex = sharedIndex+blockIdx.x*cblockSize;

    	if(globalIndex<n)
    		s_string[sharedIndex] = d_string[globalIndex];
    	else
    		s_string[sharedIndex] = '*'; //assume * not in d_string

    __syncthreads();
    	if(blockIdx.x==1)	
    		printf("%c %d %d\n",s_string[sharedIndex],sharedIndex,globalIndex);
    }


    //run the thingy in shared memory

    /*
    if (tid<n)
      {
        int beg = tid*patlen;
        int end = min (beg+(2*patlen), stringlen);
        i = beg+patlen-1;
        while (i < end) {
          int j = patlen-1;
          while (j >= 0 && (string[i] == pat[j])) {
              --i;
              --j;
          }
          if (j < 0) {
              d_retval = i+1;
              printf("\nFound at: %d\n",i+1);
              break;
          }
          i += max(delta1[j+1] , j - delta2[string[i+j]]);
        }
      }*/
}



void badCharHeuristic( char *str, int size, 
                        int badchar[NO_OF_CHARS])
{
    int i;
    for (i = 0; i < NO_OF_CHARS; i++)
         badchar[i] = -1;
 
    for (i = 0; i < size; i++)
         badchar[(int) str[i]] = i;
}
 
void goodCharHeuristic(int *shift, int *bpos,char *pat, int m)
{

    int i=m, j=m+1;
    bpos[i]=j;
 
    while(i>0)
    {
        while(j<=m && pat[i-1] != pat[j-1])
        {
            if (shift[j]==0)
                shift[j] = j-i;
 
            j = bpos[j];
        }
        i--;
        j--;
        bpos[i] = j; 
    }

    //here 

    j = bpos[0];
    for(i=0; i<=m; i++)
    {
        if(shift[i]==0)
            shift[i] = j;
 
        if (i==j)
            j = bpos[j];
    }

}


char h_string[100];
char h_pat[10];

int main(int argc, char const *argv[]){
    char *d_s, *d_p;
    int *d_d1, *d_d2;

//    cin>>h_string>>h_pat;

    for(int i=0;i<100;i++)
    	h_string[i] = 'a'+(i%26);

    for(int i=0;i<10;i++)
    	h_pat[i] = 'a' + (i%26); 


    int stringlen = strlen(h_string);
    int patlen = strlen(h_pat);
    
    int *delta1 = (int*)malloc(sizeof(int)*(patlen+1));
   
    int *bpos = (int*)malloc(sizeof(int)*(patlen+1));

    int delta2[NO_OF_CHARS];
    
   
    
    goodCharHeuristic(delta1, bpos, h_pat, patlen);
 	badCharHeuristic(h_pat, patlen, delta2);

    cudaMalloc(&d_s, stringlen*sizeof(char));
    cudaMemcpy(d_s, h_string,stringlen*sizeof(char),cudaMemcpyHostToDevice);
    cudaMalloc(&d_p, patlen*sizeof(char));
    cudaMemcpy(d_p, h_pat,patlen*sizeof(char),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d1, patlen*sizeof(char));
    cudaMemcpy(d_d1, delta1,patlen*sizeof(char),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d2, NO_OF_CHARS*sizeof(char));
    cudaMemcpy(d_d2, delta2,NO_OF_CHARS*sizeof(char),cudaMemcpyHostToDevice);

    int& n=stringlen;
    int& m=patlen;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,0);
    
    int sm_size = SHAREDMEMPERBLOCK;//devProp.sharedMemPerBlock/2; //so that atleast 2 blocks can be scheduled simultaneously
    int conceptualBlockSize = SHAREDMEMPERBLOCK- m + 1;
    int n_blocks = (n-1)/(conceptualBlockSize) + 1;//number of blocks
    int threadsPerBlock = NUMTHREADSPERBLOCK;//devProp.maxThreadsPerBlock;//max threads
    int offset = sm_size/threadsPerBlock;// number of characters each thread loads into shared mem =D
   	

    boyer_moore<<<n_blocks,threadsPerBlock>>>(d_s, n, d_p, m, d_d1, d_d2,
    															offset,conceptualBlockSize);
    
    return 0;
  }

  //things to care of 
  //1. Number of Blocks fixed to 2000
  //2.