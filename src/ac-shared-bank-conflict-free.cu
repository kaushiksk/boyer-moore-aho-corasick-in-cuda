#include "project-header.h"

texture<int, cudaTextureType2D> tex_go_to_state;
texture<unsigned int, cudaTextureType1D> tex_failure_state;
texture<unsigned int, cudaTextureType1D> tex_output_state;

__global__ void shared_kernel2 ( unsigned char *d_text, unsigned int *d_out, int m, int n, int p_size, int alphabet, int num_blocks, int sharedMemSize ) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int r, s;
	
	int i, j, column, matches = 0;
	
	int readsPerThread = sharedMemSize / ( blockDim.x * 16 );
	
	int startThread = readsPerThread * threadIdx.x;
	int stopThread = startThread + readsPerThread + ( m - 1 ) / 16 + 1;
	
	//Define space in shared memory
	//For every m - 1 multiple of 16, an additional uint4 should be reserved for redundancy
	extern __shared__ uint4 uint4_s_array[];

	//cast data to uint4
	uint4 *uint4_text = reinterpret_cast < uint4 * > ( d_text );
	uint4 uint4_var;
	
	//recast data to uchar4
	uchar4 c0, c4, c8, c12;

	//cuPrintf("start %i, stop %i\n", startThread, stopThread);
	
	for ( int globalMemIndex = blockIdx.x * sharedMemSize; globalMemIndex < n; globalMemIndex += num_blocks * sharedMemSize ) {
		
		for ( i = globalMemIndex / 16 + threadIdx.x, j = 0 + threadIdx.x; ( j < ( sharedMemSize + m - 1 ) / 16 + 1 && i < n / 16 ); i+=blockDim.x, j+=blockDim.x )
			uint4_s_array[j] = uint4_text[i];
			
		__syncthreads();
		
		r = 0;
		
		for ( column = startThread; column < stopThread && globalMemIndex + column * 16 < n; column++ ) {
			
			uint4_var = uint4_s_array[column];
			
			//recast data back to char after the memory transaction
			c0 = *reinterpret_cast<uchar4 *> ( &uint4_var.x );
			c4 = *reinterpret_cast<uchar4 *> ( &uint4_var.y );
			c8 = *reinterpret_cast<uchar4 *> ( &uint4_var.z );
			c12 = *reinterpret_cast<uchar4 *> ( &uint4_var.w );
		
			while ( ( s = tex2D ( tex_go_to_state, c0.x -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c0.y -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c0.z -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c0.w -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c4.x -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c4.y -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c4.z -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c4.w -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c8.x -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c8.y -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c8.z -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c8.w -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c12.x -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c12.y -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c12.z -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
					
			while ( ( s = tex2D ( tex_go_to_state, c12.w -'A', r ) ) == -1 )
				r = tex1Dfetch ( tex_failure_state, r );
			r = s;
					
			matches += tex1Dfetch ( tex_output_state, r );
			
		}
		
		__syncthreads();
	}
	
	d_out[idx] = matches;
}

void shared2(int m, unsigned char *text, int n, int p_size, int alphabet, int *go_to_state, unsigned int *failure_state, unsigned int *output_state ) {

	//Pointer for device memory
	int *d_go_to_state;
	unsigned int *d_failure_state, *d_output_state, *d_out;
	
	unsigned char *d_text;

	size_t pitch;
	
	int num_blocks = 30, num_threads_per_block = 256, sharedMemSize = 16128;
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
	shared_kernel2<<<dimGrid, dimBlock, sharedMemSize + 16 * ( ( m - 1 ) / 16 + 1 )>>>( d_text, d_out, m, n, p_size, alphabet, num_blocks, sharedMemSize );
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
  	
	printf ("Kernel 5 matches \t%i\t time \t%fms\n", matches, time);
			
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

	shared2( M, charText, text.size(), D, 26, go_to_table, f, out );
	return 0;
}

