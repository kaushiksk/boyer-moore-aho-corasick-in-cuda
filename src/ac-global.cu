#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include<fstream>
#include<iostream>
#include <sstream>
#include<queue>
using namespace std;

# define M 3
# define D 6



const int MAXS = M*D + 1;
 
const int MAXC = 26;
 
unsigned int out[MAXS];
 
unsigned int f[MAXS]; 

int g[MAXS][MAXC];
texture<int, cudaTextureType2D> tex_state_transition;
texture<unsigned int, cudaTextureType1D> tex_state_supply;
texture<unsigned int, cudaTextureType1D> tex_state_final;


static void checkCUDAError(const char *msg) {

	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline static void __checkCudaErrors(cudaError err, const char *file,
		const int line) {

	if (cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line,
				(int) err, cudaGetErrorString(err));
		exit(-1);
	}
}


int buildMatchingMachine(string arr[], int k)
{
    // Initialize all values in output function as 0.
    memset(out, 0, sizeof out);
 
    // Initialize all values in goto function as -1.
    memset(g, -1, sizeof g);
 
    // Initially, we just have the 0 state
    int states = 1;
 
    // Construct values for goto function, i.e., fill g[][]
    // This is same as building a Trie for arr[]
    for (int i = 0; i < k; ++i)
    {
        const string &word = arr[i];
        int currentState = 0;
 
        // Insert all characters of current word in arr[]
        for (int j = 0; j < word.size(); ++j)
        {
            int ch = word[j] - 'A';
 
            // Allocate a new node (create a new state) if a
            // node for ch doesn't exist.
            if (g[currentState][ch] == -1)
                g[currentState][ch] = states++;
 
            currentState = g[currentState][ch];
        }
 
        // Add current word in output function
        out[currentState] += 1;
    }
 
    // For all characters which don't have an edge from
    // root (or state 0) in Trie, add a goto edge to state
    // 0 itself
    for (int ch = 0; ch < MAXC; ++ch)
        if (g[0][ch] == -1)
            g[0][ch] = 0;
 
    // Now, let's build the failure function
 
    // Initialize values in fail function
    memset(f, -1, sizeof f);
 
    // Failure function is computed in breadth first order
    // using a queue
    queue<int> q;
 
     // Iterate over every possible input
    for (int ch = 0; ch < MAXC; ++ch)
    {
        // All nodes of depth 1 have failure function value
        // as 0. For example, in above diagram we move to 0
        // from states 1 and 3.
        if (g[0][ch] != 0)
        {
            f[g[0][ch]] = 0;
            q.push(g[0][ch]);
        }
    }
 
    // Now queue has states 1 and 3
    while (q.size())
    {
        // Remove the front state from queue
        int state = q.front();
        q.pop();
 
        // For the removed state, find failure function for
        // all those characters for which goto function is
        // not defined.
        for (int ch = 0; ch <= MAXC; ++ch)
        {
            // If goto function is defined for character 'ch'
            // and 'state'
            if (g[state][ch] != -1)
            {
                // Find failure state of removed state
                int failure = f[state];
 
                // Find the deepest node labeled by proper
                // suffix of string from root to current
                // state.
                while (g[failure][ch] == -1)
                      failure = f[failure];
 
                failure = g[failure][ch];
                f[g[state][ch]] = failure;
 
                // Merge output values
                out[g[state][ch]] += out[failure];
 
                // Insert the next level node (of Trie) in Queue
                q.push(g[state][ch]);
            }
        }
    }
 
    return states;
}



__global__ void ac_kernel1 ( int *d_state_transition, unsigned int *d_state_supply, unsigned int *d_state_final, unsigned char *d_text, unsigned int *d_out, size_t pitch, int m, int n, int p_size, int alphabet, int numBlocks ) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int effective_pitch = pitch / sizeof ( int );
	
	int charactersPerBlock = n / numBlocks;
	
	int startBlock = blockIdx.x * charactersPerBlock;
	int stopBlock = startBlock + charactersPerBlock;
	
	int charactersPerThread = ( stopBlock - startBlock ) / blockDim.x;
	
	int startThread = startBlock + charactersPerThread * threadIdx.x;
	int stopThread;
	if( blockIdx.x == numBlocks -1 && threadIdx.x==blockDim.x-1)
		stopThread = n - 1;	
	else stopThread = startThread + charactersPerThread + m-1;

	int r = 0, s;
	
	int column;
	
	//cuPrintf("Working from %i to %i chars %i\n", startThread, stopThread, charactersPerThread);
	
	for ( column = startThread; ( column < stopThread && column < n ); column++ ) {

		while ( ( s = d_state_transition[r * effective_pitch + (d_text[column]-(unsigned char)'A')] ) == -1 )
			r = d_state_supply[r];
		r = s;
			
		d_out[idx] += d_state_final[r];
	}
}
__global__ void ac_kernel2 ( unsigned char *d_text, unsigned int *d_out, int m, int n, int p_size, int alphabet, int numBlocks ) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int charactersPerBlock = n / numBlocks;
	
	int startBlock = blockIdx.x * charactersPerBlock;
	int stopBlock = startBlock + charactersPerBlock;
	
	int charactersPerThread = ( stopBlock - startBlock ) / blockDim.x;
	
	int startThread = startBlock + charactersPerThread * threadIdx.x;
	int stopThread = startThread + charactersPerThread + m - 1;
	
	int r = 0, s;
	
	int column;
	
	for ( column = startThread; ( column < stopThread && column < n ); column++ ) {

		while ( ( s = tex2D ( tex_state_transition, (d_text[column]-(unsigned char)'A'), r ) ) == -1 )
			r = tex1Dfetch ( tex_state_supply, r );
		r = s;
			
		d_out[idx] += tex1Dfetch ( tex_state_final, r );
	}
}



void cuda_ac1 ( int m, unsigned char *text, int n, int p_size, int alphabet, int *state_transition, unsigned int *state_supply, unsigned int *state_final ) {

	//Pointer for device memory
	int *d_state_transition;
	unsigned int *d_state_supply, *d_state_final, *d_out;
	
	unsigned char *d_text;

	size_t pitch;
	
	int numBlocks = 8, numThreadsPerBlock = 1024;

	dim3 dimGrid ( numBlocks );
	dim3 dimBlock ( numThreadsPerBlock );
	
	if ( n < numBlocks * numThreadsPerBlock * m ) {
		printf("The text size is too small\n");
		exit(1);
	}
	
	//Allocate host memory for results array
	unsigned int *h_out = ( unsigned int * ) malloc ( numBlocks * numThreadsPerBlock * sizeof ( unsigned int ) );
	memset ( h_out, 0, numBlocks * numThreadsPerBlock * sizeof ( unsigned int ) );

	//Allocate 1D device memory
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_text, n * sizeof ( unsigned char ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_state_supply, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_state_final, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_out, numBlocks * numThreadsPerBlock * sizeof ( unsigned int ) ) );
	
	//Allocate 2D device memory
	checkCudaErrors ( cudaMallocPitch ( &d_state_transition, &pitch, alphabet * sizeof ( int ), ( m * p_size + 1 ) ) );
	
	//Copy 1D host memory to device
	checkCudaErrors ( cudaMemcpy ( d_text, text, n * sizeof ( unsigned char ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_state_supply, state_supply, ( m * p_size + 1 ) * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_state_final, state_final, ( m * p_size + 1 ) * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_out, h_out, numBlocks * numThreadsPerBlock * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );

	//Copy 2D host memory to device
	checkCudaErrors ( cudaMemcpy2D ( d_state_transition, pitch, state_transition, alphabet * sizeof ( int ), alphabet * sizeof ( int ), ( m * p_size + 1 ), cudaMemcpyHostToDevice ) );
	
	//Create timer
	cudaEvent_t start, stop;

	float time;
	
	//Create the timer events
	cudaEventCreate ( &start );
	cudaEventCreate ( &stop );
	
	//Start the event clock	
	cudaEventRecord ( start, 0 );
	
	//cudaPrintfInit();
	
	//Executing kernel in the device
	ac_kernel1<<<dimGrid, dimBlock>>>( d_state_transition, d_state_supply, d_state_final, d_text, d_out, pitch, m, n, p_size, alphabet, numBlocks );
	checkCUDAError("kernel invocation");
	
	//cudaPrintfDisplay(stdout, true);
	//cudaPrintfEnd();

	cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	
	cudaEventElapsedTime ( &time, start, stop );
	
	cudaEventDestroy ( start );
	cudaEventDestroy ( stop );
	
	//Get back the results from the device
	cudaMemcpy ( h_out, d_out, numBlocks * numThreadsPerBlock * sizeof ( unsigned int ), cudaMemcpyDeviceToHost );
	
  	//Look at the results
  	int i, matches = 0;
  	for ( i = 0; i < numBlocks * numThreadsPerBlock; i++ )
  		matches += h_out[i];
  		
	printf ("Global Memory Kernel 1 matches \t%i\t time \t%fms\n", matches, time);
		
	//Free host and device memory
	free ( h_out );
	
	cudaFree ( d_text );
	cudaFree ( d_state_transition );
	cudaFree ( d_state_supply );
	cudaFree ( d_state_final );
	cudaFree ( d_out );
}


void cuda_ac2 ( int m, unsigned char *text, int n, int p_size, int alphabet, int *state_transition, unsigned int *state_supply, unsigned int *state_final ) {


	//Pointer for device memory
	int *d_state_transition;
	unsigned int *d_state_supply, *d_state_final, *d_out;
	
	unsigned char *d_text;

	size_t pitch;
	
	int numBlocks = 8, numThreadsPerBlock = 1024;
	dim3 dimGrid ( numBlocks );
	dim3 dimBlock ( numThreadsPerBlock );
	
	if ( n < numBlocks * numThreadsPerBlock * m ) {
		printf("The text size is too small\n");
		exit(1);
	}
	
	//Allocate host memory for results array
	unsigned int *h_out = ( unsigned int * ) malloc ( numBlocks * numThreadsPerBlock * sizeof ( unsigned int ) );
	memset ( h_out, 0, numBlocks * numThreadsPerBlock * sizeof ( unsigned int ) );

	//Allocate 1D device memory
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_text, n * sizeof ( unsigned char ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_state_supply, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_state_final, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_out, numBlocks * numThreadsPerBlock * sizeof ( unsigned int ) ) );
	
	//Allocate 2D device memory
	checkCudaErrors ( cudaMallocPitch ( &d_state_transition, &pitch, alphabet * sizeof ( int ), ( m * p_size + 1 ) ) );
	
	//Copy 1D host memory to device
	checkCudaErrors ( cudaMemcpy ( d_text, text, n * sizeof ( unsigned char ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_state_supply, state_supply, ( m * p_size + 1 ) * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_state_final, state_final, ( m * p_size + 1 ) * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_out, h_out, numBlocks * numThreadsPerBlock * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );

	//Copy 2D host memory to device
	checkCudaErrors ( cudaMemcpy2D ( d_state_transition, pitch, state_transition, alphabet * sizeof ( int ), alphabet * sizeof ( int ), ( m * p_size + 1 ), cudaMemcpyHostToDevice ) );
	
	//Bind the preprocessing tables to the texture cache
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	
	checkCudaErrors ( cudaBindTexture2D ( 0, tex_state_transition, d_state_transition, desc, alphabet, m * p_size + 1, pitch ) );
	checkCudaErrors ( cudaBindTexture ( 0, tex_state_supply, d_state_supply, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaBindTexture ( 0, tex_state_final, d_state_final, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	
	//Create timer
	cudaEvent_t start, stop;

	float time;

	//Create the timer events
	cudaEventCreate ( &start );
	cudaEventCreate ( &stop );
	
	//Start the event clock	
	cudaEventRecord ( start, 0 );
	
	//Executing kernel in the device
	ac_kernel2<<<dimGrid, dimBlock>>>( d_text, d_out, m, n, p_size, alphabet, numBlocks );
	checkCUDAError("kernel invocation");
	
	cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	
	cudaEventElapsedTime ( &time, start, stop );
	
	cudaEventDestroy ( start );
	cudaEventDestroy ( stop );

	//Get back the results from the device
	cudaMemcpy ( h_out, d_out, numBlocks * numThreadsPerBlock * sizeof ( unsigned int ), cudaMemcpyDeviceToHost );
   
  	//Look at the results
  	int i, matches = 0;
  	for ( i = 0; i < numBlocks * numThreadsPerBlock; i++ )
  		matches += h_out[i];
  	
	printf ("Texture Memory Kernel matches \t%i\t time \t%fms\n", matches, time);
	
	cudaUnbindTexture ( tex_state_transition );
	cudaUnbindTexture ( tex_state_supply );
	cudaUnbindTexture ( tex_state_final );
	
	//Free host and device memory
	free ( h_out );
	
	cudaFree ( d_text );
	cudaFree ( d_state_transition );
	cudaFree ( d_state_supply );
	cudaFree ( d_state_final );
	cudaFree ( d_out );
}


int main(){

	string patterns[]={"ATC","GTG","GTC","ATG","CAA","ATT"};
	int k = sizeof(patterns)/sizeof(patterns[0]);
	string text;

	std::ifstream t("data.txt");
	std::stringstream buffer;
	buffer << t.rdbuf();	
	text = buffer.str();
	unsigned char *charText = (unsigned char*)text.c_str();
	
	buildMatchingMachine(patterns, k);

	int *goToTable = (int*)malloc(sizeof(int)*MAXC*MAXS);
	for(int i=0;i<MAXS;i++)
		for(int j=0;j<MAXC;j++)
			goToTable[i*MAXC+j] = g[i][j];

//(int m, unsigned char *text, int n, int p_size, int alphabet, int *state_transition, unsigned int *state_supply, unsigned int *state_final ) {
	
	cuda_ac1(M,charText,text.size(),D,26,goToTable,f,out);
	//cuda_ac2(M,charText,text.size(),D,26,goToTable,f,out);
	
	return 0;
}




