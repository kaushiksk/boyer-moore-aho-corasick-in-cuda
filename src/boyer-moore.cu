#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#define ALPHABET_LEN 4
#define NOT_FOUND patlen
#define max(a, b) ((a < b) ? b : a)


void make_delta1(int *delta1, int8_t *pat, int32_t patlen) {
    int i;

    for (i=0; i < ALPHABET_LEN; i++) {
        delta1[i] = NOT_FOUND;
    }
    for (i=0; i < patlen-1; i++) {
        delta1[pat[i]] = patlen-1 - i;
    }
}

int is_prefix(int8_t *word, int wordlen, int pos) {
    int i;
    int suffixlen = wordlen - pos;

    for (i=0; i < suffixlen; i++) {
        if (word[i] != word[pos+i]) {
            return 0;
        }
    }
    return 1;
}

int suffix_length(int8_t *word, int wordlen, int pos) {
    int i;
    for (i = 0; (word[pos-i] == word[wordlen-1-i]) && (i < pos); i++);
    return i;
}

void make_delta2(int *delta2, int8_t *pat, int32_t patlen) {
    int p;
    int last_prefix_index = 1;

    for (p=patlen-1; p>=0; p--) {
        if (is_prefix(pat, patlen, p+1)) {
            last_prefix_index = p+1;
        }
        delta2[p] = (patlen-1 - p) + last_prefix_index;
    }

    for (p=0; p < patlen-1; p++) {
        int slen = suffix_length(pat, patlen, p);
        if (pat[p - slen] != pat[patlen-1 - slen]) {
            delta2[patlen-1 - slen] = patlen-1 - p + slen;
        }
    }
}

__device__ int d_retval;

__global__ void boyer_moore (int8_t *string, int32_t stringlen, int8_t *pat, int32_t patlen, int *delta1, int *delta2, int n) {
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
              break;
          }
          i += max(delta1[string[i]], delta2[j]);
        }
      }
}


int8_t h_string[1000000];
int8_t h_pat[100];

int main(int argc, char const *argv[]) {
    int8_t *d_s, *d_p;
    int *d_d1, *d_d2;
    int32_t strlen = 1000000;
    int32_t patlen = 100;
    srand(time(NULL));
    int i;
    char con [] = "ACGT";
    for(i=0;i<strlen;i++)
      h_string[i] = rand ()%4;

    int patid = rand ()%10000;
  
    for(i=0;i<patlen;i++)
      h_pat[i] = h_string[patid++];

    printf("The String is: ");
    for (i=0;i<strlen;i++)
      printf("%c", con[h_string[i]]);
    printf("\nThe search keyword is: ");
    for (i=0;i<patlen;i++)
      printf("%c", con[h_pat[i]]);

    int delta1[ALPHABET_LEN];
    int delta2[patlen];
    make_delta1(delta1, h_pat, patlen);
    make_delta2(delta2, h_pat, patlen);


    cudaMalloc(&d_s, strlen*sizeof(int8_t));
    cudaMemcpy(d_s, h_string,strlen*sizeof(int8_t),cudaMemcpyHostToDevice);
    cudaMalloc(&d_p, patlen*sizeof(int8_t));
    cudaMemcpy(d_p, h_pat,patlen*sizeof(int8_t),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d1, ALPHABET_LEN*sizeof(int));
    cudaMemcpy(d_d1, delta1,ALPHABET_LEN*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_d2, patlen*sizeof(int));
    cudaMemcpy(d_d2, delta2,patlen*sizeof(int),cudaMemcpyHostToDevice);

    int n = strlen/patlen;
    
    int block_size = 1024;
    int n_blocks = n/block_size + (n%block_size==0?0:1);
    boyer_moore<<<n_blocks,block_size>>>(d_s, strlen, d_p, patlen, d_d1, d_d2, n);
    cudaDeviceSynchronize();
    int answer;
    cudaMemcpyFromSymbol(&answer, d_retval, sizeof(int), 0, cudaMemcpyDeviceToHost);

    printf("\nString found at %d\n", answer);
    return 0;
}
