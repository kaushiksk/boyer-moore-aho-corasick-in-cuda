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
 


__global__ void boyer_moore (char *d_string, int n, char *pat, int m, const int * __restrict__ delta1, const int * __restrict__ delta2, int t_size){
    int i;
    int tidid = blockIdx.x*blockDim.x+threadIdx.x;//0,1,2,....

    int stringid = tid*(t_size-m+1);//0,(t_size-m+1),.....


    __shared__ char s_string[];
    //load to shared memory
    for(i=0;i<t_size;i++){
      if(stringid+i<n)
        s_string[i] = d_string[stringid+i]
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


char h_string[1000000];
char h_pat[100];

int main(int argc, char const *argv[]){
    char *d_s, *d_p;
    int *d_d1, *d_d2;

    cin>>h_string>>h_pat;

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

    int sm_size = devProp.sharedMemPerBlock/2; //so that atleast 2 blocks can be scheduled simultaneously
    int n_blocks = n/(sm_size-(m-1)) + (n%(sm_size-(m-1))==0?0:1);//number of blocks
    int b_size = devProp.maxThreadsPerBlock;//max threads
    int t_size = sm_size/b_size + m-1;// number of characters each thread compares =D
    //int n_blocks = n/block_size + (n%block_size==0?0:1);
    boyer_moore<<<n_blocks,block_size,sm_size*sizeof(char)>>>(d_s, n, d_p, m, d_d1, d_d2,t_size);
    
    return 0;
  }
