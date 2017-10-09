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
 
    // Initialize all occurrences as -1
    for (i = 0; i < NO_OF_CHARS; i++)
         badchar[i] = -1;
 
    // Fill the actual value of last occurrence 
    // of a character
    for (i = 0; i < size; i++)
         badchar[(int) str[i]] = i;
}
 

// preprocessing for strong good suffix rule
void preprocess_strong_suffix(int *shift, int *bpos,char *pat, int m)
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
        i--;
        j--;
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

void search( char *text,  char *pat)
{
 
    int badchar[NO_OF_CHARS];
    int n = strlen(text);
 
    int *bpos, *shift;
    bpos = (int*)malloc(sizeof(int)*(m+1));
    shift = (int*)malloc(sizeof(int)*(m+1));

 
    //initialize all occurence of shift to 0
    for(int i=0;i<m+1;i++) shift[i]=0;
 
    //do preprocessing
    preprocess_strong_suffix(shift, bpos, pat, m);
    preprocess_case2(shift, bpos, pat, m);
 
    /* Fill the bad character array by calling 
       the preprocessing function badCharHeuristic() 
       for given pattern */
    badCharHeuristic(pat, m, badchar);
 
    int s = 0;  // s is shift of the pattern with 
                // respect to text
    while(s <= (n - m))
    {
        int j = m-1;
 
        /* Keep reducing index j of pattern while 
           characters of pattern and text are 
           matching at this shift s */
        while(j >= 0 && pat[j] == text[s+j])
            j--;
 
        /* If the pattern is present at current
           shift, then index j will become -1 after
           the above loop */
        if (j < 0)
        {
            printf("\n pattern occurs at shift = %d", s);
 
            /* Shift the pattern so that the next 
               character in text aligns with the last 
               occurrence of it in pattern.
               The condition s+m < n is necessary for 
               the case when pattern occurs at the end 
               of text */
            s += shift[0];
 
        }
 
        else

            s += max(shift[j+1] , j - badchar[text[s+j]]);
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
        i = beg+patlen-1;
        while (i < end) {
          int j = patlen-1;
          while (j >= 0 && (string[i] == pat[j])) {
              //printf("here in loop\n");
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
   
    int *bpos = (int*)malloc(sizeof(int)*(patlen+1));

    int delta2[NO_OF_CHARS];
    
    /*
    make_delta1(delta1, h_pat, patlen);
    make_delta2(delta2, h_pat, patlen);
	*/
    
    preprocess_strong_suffix(delta1, bpos, h_pat, patlen);
    preprocess_case2(delta1, bpos, h_pat, patlen);
 	badCharHeuristic(h_pat, patlen, delta2);

    cudaMalloc(&d_s, stringlen*sizeof(char));
    cudaMemcpy(d_s, h_string,stringlen*sizeof(char),cudaMemcpyHostToDevice);
    cudaMalloc(&d_p, patlen*sizeof(char));
    cudaMemcpy(d_p, h_pat,patlen*sizeof(char),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d1, patlen*sizeof(char));
    cudaMemcpy(d_d1, delta1,patlen*sizeof(char),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d2, NO_OF_CHARS*sizeof(char));
    cudaMemcpy(d_d2, delta2,NO_OF_CHARS*sizeof(char),cudaMemcpyHostToDevice);

    int n = stringlen/patlen;
    
    int block_size = 1024;
    int n_blocks = n/block_size + (n%block_size==0?0:1);
    boyer_moore<<<n_blocks,block_size>>>(d_s, stringlen, d_p, patlen, d_d1, d_d2, n);
    
    //cudaDeviceSynchronize();
    //int answer;
    //cudaMemcpyFromSymbol(&answer, d_retval, sizeof(int), 0, cudaMemcpyDeviceToHost);

    //printf("\nString found at %d\n", answer);
    return 0;
}
