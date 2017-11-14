#include "project-header.h"

__global__ void global_memory_kernel(int *d_go_to_state, unsigned int *d_failure_state, unsigned int *d_output_state, unsigned char *d_text, unsigned int *d_out, size_t pitch, int m, int n, int p_size, int alphabet, int num_blocks ) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int effective_pitch = pitch / sizeof ( int );
	
	int chars_per_block = n / num_blocks;
	
	int start_block = blockIdx.x * chars_per_block;
	int stop_block = start_block + chars_per_block;
	
	int chars_per_thread = ( stop_block - start_block ) / blockDim.x;
	
	int start_thread = start_block + chars_per_thread * threadIdx.x;
	int stop_thread;
	if( blockIdx.x == num_blocks -1 && threadIdx.x==blockDim.x-1)
		stop_thread = n - 1;	
	else stop_thread = start_thread + chars_per_thread + m-1;

	int r = 0, s;
	
	int column;
	
	for ( column = start_thread; ( column < stop_thread && column < n ); column++ ) {

		while ( ( s = d_go_to_state[r * effective_pitch + (d_text[column]-(unsigned char)'A')] ) == -1 )
			r = d_failure_state[r];
		r = s;
			
		d_out[idx] += d_output_state[r];
	}
}




void global_memory_wrapper_func ( int m, unsigned char *text, int n, int p_size, int alphabet, int *go_to_state, unsigned int *failure_state, unsigned int *output_state ) {

	//Pointer for device memory
	int *d_go_to_state;
	unsigned int *d_failure_state, *d_output_state, *d_out;
	
	unsigned char *d_text;

	size_t pitch;
	
	int num_blocks = 8, num_threads_per_block = 1024;

	dim3 dimGrid ( num_blocks );
	dim3 dimBlock ( num_threads_per_block );
	
	if ( n < num_blocks * num_threads_per_block * m ) {
		printf("The text size is too small\n");
		exit(1);
	}
	
	//Allocate host memory for results array
	unsigned int *h_out = ( unsigned int * ) malloc ( num_blocks * num_threads_per_block * sizeof ( unsigned int ) );
	memset ( h_out, 0, num_blocks * num_threads_per_block * sizeof ( unsigned int ) );

	//Allocate 1D device memory
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_text, n * sizeof ( unsigned char ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_failure_state, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_output_state, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_out, num_blocks * num_threads_per_block * sizeof ( unsigned int ) ) );
	
	//Allocate 2D device memory
	checkCudaErrors ( cudaMallocPitch ( &d_go_to_state, &pitch, alphabet * sizeof ( int ), ( m * p_size + 1 ) ) );
	
	//Copy 1D host memory to device
	checkCudaErrors ( cudaMemcpy ( d_text, text, n * sizeof ( unsigned char ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_failure_state, failure_state, ( m * p_size + 1 ) * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_output_state, output_state, ( m * p_size + 1 ) * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_out, h_out, num_blocks * num_threads_per_block * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );

	//Copy 2D host memory to device
	checkCudaErrors ( cudaMemcpy2D ( d_go_to_state, pitch, go_to_state, alphabet * sizeof ( int ), alphabet * sizeof ( int ), ( m * p_size + 1 ), cudaMemcpyHostToDevice ) );
	
	//Create timer
	cudaEvent_t start, stop;

	float time;
	
	//Create the timer events
	cudaEventCreate ( &start );
	cudaEventCreate ( &stop );
	
	//Start the event clock	
	cudaEventRecord ( start, 0 );
	
	
	
	//Executing kernel in the device
	global_memory_kernel<<<dimGrid, dimBlock>>>( d_go_to_state, d_failure_state, d_output_state, d_text, d_out, pitch, m, n, p_size, alphabet, num_blocks );
	checkCUDAError("kernel invocation");
	

	cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	
	cudaEventElapsedTime ( &time, start, stop );
	
	cudaEventDestroy ( start );
	cudaEventDestroy ( stop );
	
	//Get back the results from the device
	cudaMemcpy ( h_out, d_out, num_blocks * num_threads_per_block * sizeof ( unsigned int ), cudaMemcpyDeviceToHost );
	
  	//Look at the results
  	int i, matches = 0;
  	for ( i = 0; i < num_blocks * num_threads_per_block; i++ )
  		matches += h_out[i];
  		
	printf ("Global Memory Kernel 1 matches \t%i\t time \t%fms\n", matches, time);
		
	//Free host and device memory
	free ( h_out );
	
	cudaFree ( d_text );
	cudaFree ( d_go_to_state );
	cudaFree ( d_failure_state );
	cudaFree ( d_output_state );
	cudaFree ( d_out );
}




int main(){

 	string patterns[]={"ATC","GTG","GTC","ATG","CAA","ATT"};
    int k = sizeof( patterns )/sizeof( patterns[0] );
    string text;

    ifstream t( "data.txt" );
    stringstream buffer;
    buffer << t.rdbuf();    
    text = buffer.str();
    unsigned char *charText = ( unsigned char* )text.c_str();
    
    buildMatchingMachine(patterns, k);

    int *go_to_table = (int*)malloc( sizeof(int)*MAXC*MAXS );
    
    for(int i=0;i<MAXS;i++)
        for(int j=0;j<MAXC;j++)
            go_to_table[i*MAXC+j] = g[i][j];
	
	global_memory_wrapper_func( M, charText, text.size(), D, 26, go_to_table, f, out );
	
	return 0;
}




