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
 
// A utility function to get maximum of two integers
 
// The preprocessing function for Boyer Moore's
// bad character heuristic
void badCharHeuristic( char *str, int size, 
                        int badchar[NO_OF_CHARS])
{
    int i;
    for (i = 0; i < NO_OF_CHARS; i++)
         badchar[i] = -1;
 
    for (i = 0; i < size; i++)
         badchar[(int) str[i]] = i;
}
 
// preprocessing for strong good suffix rule
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

//-----------------------------------------------------------------------------------------------

__device__ int d_retval;

__global__ void boyer_moore (char *string, int stringlen, char *pat, int patlen, int *delta1, int *delta2, int n) {
    int i;
    d_retval = -1;
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    __syncthreads();
    if (tid<n)
      {

        int beg = tid*patlen;
        int end = min (beg+(2*patlen), stringlen);

        //printf("%d %d %d \n",tid,beg,end);
        i = beg;
        while (i < end) {

          int j = patlen-1;
          while (j >= 0 && (string[i+j] == pat[j])) {
              //printf("here in loop\n");
              --j;
          }

          if(threadIdx.x==0&&blockIdx.x==0)
            printf("%d %d %d %d \n",j+1,i+j,delta1[j+1],j - delta2[string[i+j]]);

          if (j < 0) {
              d_retval = i+1;
              printf("\nFound at: %d\n",i+1);
              break;
          }
          else
            i += max(delta1[j+1] , j - delta2[string[i+j]]);
         
        }
      }
}


char h_string[1000000];
char h_pat[100];

int main(int argc, char const *argv[]) {
    char *d_s, *d_p;
    int *d_d1, *d_d2;

    cin>>h_string>>h_pat;

    int stringlen = strlen(h_string);
    int patlen = strlen(h_pat);
    
    int *delta1 = (int*)malloc(sizeof(int)*(patlen+1));
    for(int i=0;i<patlen+1;i++) delta1[i]=0;
    int *bpos = (int*)malloc(sizeof(int)*(patlen+1));

    int delta2[NO_OF_CHARS];
    
   
    
    preprocess_strong_suffix(delta1, bpos, h_pat, patlen);
    preprocess_case2(delta1, bpos, h_pat, patlen);
 	  badCharHeuristic(h_pat, patlen, delta2);

    for(int i=0;i<patlen+1;i++)
      printf("%d ",delta1[i]);
    cout<<endl;
    for(int i=0;i<NO_OF_CHARS;i++)
      printf("%d ",delta2[i]);
    cout<<endl;
    cudaMalloc(&d_s, stringlen*sizeof(char));
    cudaMemcpy(d_s, h_string,stringlen*sizeof(char),cudaMemcpyHostToDevice);
    cudaMalloc(&d_p, patlen*sizeof(char));
    cudaMemcpy(d_p, h_pat,patlen*sizeof(char),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d1, (patlen+1)*sizeof(int));
    cudaMemcpy(d_d1, delta1,(patlen+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d2, NO_OF_CHARS*sizeof(int));
    cudaMemcpy(d_d2, delta2,NO_OF_CHARS*sizeof(int),cudaMemcpyHostToDevice);

    int n = stringlen/patlen;
    
    int block_size = 1024;
    int n_blocks = n/block_size + (n%block_size==0?0:1);
    boyer_moore<<<n_blocks,block_size>>>(d_s, stringlen, d_p, patlen, d_d1, d_d2, n);
    
    return 0;
}
