#include "project-header.h"

texture<int, cudaTextureType2D> tex_go_to_state;
texture<unsigned int, cudaTextureType1D> tex_failure_state;
texture<unsigned int, cudaTextureType1D> tex_output_state;


__global__ void shared_kernel1 ( unsigned char *d_text, unsigned int *d_out, int m, int n, int p_size, int alphabet, int num_blocks, int sharedMemSize ) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int r, s;
	
	int i, j, column;
	
	int chars_per_thread = sharedMemSize / blockDim.x;
	
	int start_thread = chars_per_thread * threadIdx.x;
	int stop_thread = start_thread + chars_per_thread + m - 1;

	//Define space in shared memory
	extern __shared__ unsigned char s_array[];
	
	for ( int globalMemIndex = blockIdx.x * sharedMemSize; globalMemIndex < n; globalMemIndex += num_blocks * sharedMemSize ) {
	
		/*if ( threadIdx.x == 0 )
			for ( i = globalMemIndex, j = 0; ( j < sharedMemSize + m - 1 && i < n ); i++, j++ )
				s_array[j] = d_text[i];
		*/
		
		for ( i = globalMemIndex + threadIdx.x, j = 0 + threadIdx.x; ( j < sharedMemSize + m - 1 && i < n ); i+=blockDim.x, j+=blockDim.x )
			s_array[j] = d_text[i];
			
		__syncthreads();
		
		r = 0;
		
		for ( column = start_thread; ( column < stop_thread && globalMemIndex + column < n ); column++ ) {
		
			while ( ( s = tex2D ( tex_go_to_state, s_array[column]-'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
			
			d_out[idx] += tex1Dfetch ( tex_output_state, r );
		}
		
		__syncthreads();
	}
}


void shared1 ( int m, unsigned char *text, int n, int p_size, int alphabet, int *go_to_state, unsigned int *failure_state, unsigned int *output_state ) {

	//Pointer for device memory
	int *d_go_to_state;
	unsigned int *d_failure_state, *d_output_state, *d_out;
	
	unsigned char *d_text;

	size_t pitch;
	
	int num_blocks = 24, num_threads_per_block = 1024, sharedMemSize = 16384;
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
	
	//Bind the preprocessing tables to the texture cache
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
	checkCudaErrors ( cudaBindTexture2D ( 0, tex_go_to_state, d_go_to_state, desc, alphabet, m * p_size + 1, pitch ) );
	checkCudaErrors ( cudaBindTexture ( 0, tex_failure_state, d_failure_state, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaBindTexture ( 0, tex_output_state, d_output_state, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	
	//Create timer
	cudaEvent_t start, stop;

	float time;

	//Create the timer events
	cudaEventCreate ( &start );
	cudaEventCreate ( &stop );
	
	//Start the event clock	
	cudaEventRecord ( start, 0 );
	
	//Executing kernel in the device
	shared_kernel1<<<dimGrid, dimBlock, sharedMemSize + m - 1>>>( d_text, d_out, m, n, p_size, alphabet, num_blocks, sharedMemSize );
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
  	
	printf ("Shared Memory Kernel 1 matches \t%i\t time \t%fms\n", matches, time);
	
	cudaUnbindTexture ( tex_go_to_state );
	cudaUnbindTexture ( tex_failure_state );
	cudaUnbindTexture ( tex_output_state );
	
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

	shared1( M, charText, text.size(), D, 26, go_to_table, f, out );
	return 0;
}

