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
   	int gid = cblockSize*blockIdx.x;
    __shared__ char s_string[SHAREDMEMPERBLOCK];

    int idx = threadIdx.x;
    int sharedIndex;
    int globalIndex;

    for(i=0;i<offset;i++){
    	sharedIndex = idx + i*blockDim.x;
    	globalIndex = sharedIndex+blockIdx.x*cblockSize;

    	if(globalIndex<n)
    		s_string[sharedIndex] = d_string[globalIndex];
    	else
    		s_string[sharedIndex] = '*'; //assume * not in d_string
    
   	}
   	__syncthreads();


        int beg = idx*offset;
        int end = min (beg+offset+m, SHAREDMEMPERBLOCK);
        

        i = beg;
        while (i < end) {
          int j = m-1;

          while (j >= 0&&(s_string[i+j] == pat[j])) {
              --j;
              
              }
          if (j < 0) {
              
              printf("\nFound at: %d %d %d \n",gid+i+1,beg,end);
              break;
          }

          
          i += max(delta1[j+1] , j - delta2[s_string[i+j]]);
        }

    

      
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
 
void preprocess_strong_suffix(int *shift, int *bpos,
                                char *pat, int m)
{
    // m is the length of pattern 
    int i=m, j=m+1;
    bpos[i]=j;
 
    while(i>0)
    {
        /*if character at position i-1 is not equivalent to
          character at j-1, then continue searching to right
          of the pattern for border */
        while(j<=m && pat[i-1] != pat[j-1])
        {
            /* the character preceding the occurence of t in 
               pattern P is different than mismatching character in P, 
               we stop skipping the occurences and shift the pattern
               from i to j */
            if (shift[j]==0)
                shift[j] = j-i;
 
            //Update the position of next border 
            j = bpos[j];
        }
        /* p[i-1] matched with p[j-1], border is found.
           store the  beginning position of border */
        i--;j--;
        bpos[i] = j; 
    }
}
 
//Preprocessing for case 2
void preprocess_case2(int *shift, int *bpos,
                      char *pat, int m)
{
    int i, j;
    j = bpos[0];
    for(i=0; i<=m; i++)
    {
        /* set the border postion of first character of pattern
           to all indices in array shift having shift[i] = 0 */
        if(shift[i]==0)
            shift[i] = j;
 
        /* suffix become shorter than bpos[0], use the position of 
           next widest border as value of j */
        if (i==j)
            j = bpos[j];
    }
}

char h_string[100];
char h_pat[10];

int main(int argc, char const *argv[]){
    char *d_s, *d_p;
    int *d_d1, *d_d2;

    //cin>>h_string>>h_pat;

    for(int i=0;i<100;i++)
    	h_string[i] = 'a'+(i%26);

    for(int i=0;i<10;i++)
    	h_pat[i] = 'a' + (i%26); 
	

    int stringlen = strlen(h_string);
    int patlen = strlen(h_pat);
    
    int *delta1 = (int*)malloc(sizeof(int)*(patlen+1));for(int i=0;i<patlen+1;i++) delta1[i]=0;
   
    int *bpos = (int*)malloc(sizeof(int)*(patlen+1));

    int delta2[NO_OF_CHARS];
    
   	cout<<h_string<<" "<<h_pat<<endl;
    
    preprocess_strong_suffix(delta1, bpos, h_pat, patlen);
    preprocess_case2(delta1, bpos, h_pat, patlen);
 	badCharHeuristic(h_pat, patlen, delta2);

    cudaMalloc(&d_s, stringlen*sizeof(char));
    cudaMemcpy(d_s, h_string,stringlen*sizeof(char),cudaMemcpyHostToDevice);
    cudaMalloc(&d_p, patlen*sizeof(char));
    cudaMemcpy(d_p, h_pat,patlen*sizeof(char),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d1, (patlen+1)*sizeof(int));
    cudaMemcpy(d_d1, delta1,(patlen+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d2, NO_OF_CHARS*sizeof(int));
    cudaMemcpy(d_d2, delta2,NO_OF_CHARS*sizeof(int),cudaMemcpyHostToDevice);

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