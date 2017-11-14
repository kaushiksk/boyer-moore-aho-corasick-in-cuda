#include "project-header-bits.h"

texture<int, cudaTextureType2D> tex_state_transition;
texture<unsigned int, cudaTextureType1D> tex_state_supply;
texture<unsigned int, cudaTextureType1D> tex_state_final;


__global__ void texture_memory_kernel( unsigned char *d_text, unsigned int *d_out, int m, int n, int p_size, int alphabet, int numBlocks ) {


	
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
			
		d_out[column] = tex1Dfetch ( tex_state_final, r );
	}
}





void texture_memory_kernel_wrapper ( int m, unsigned char *text, int n, int p_size, int alphabet, int *state_transition, unsigned int *state_supply, unsigned int *state_final ) {


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
	unsigned int *h_out = ( unsigned int * ) malloc ( n * sizeof ( unsigned int ) );
	memset ( h_out, 0, n * sizeof ( unsigned int ) );

	//Allocate 1D device memory
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_text, n * sizeof ( unsigned char ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_state_supply, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_state_final, ( m * p_size + 1 ) * sizeof ( unsigned int ) ) );
	checkCudaErrors ( cudaMalloc ( ( void** ) &d_out, n * sizeof ( unsigned int ) ) );
	
	//Allocate 2D device memory
	checkCudaErrors ( cudaMallocPitch ( &d_state_transition, &pitch, alphabet * sizeof ( int ), ( m * p_size + 1 ) ) );
	
	//Copy 1D host memory to device
	checkCudaErrors ( cudaMemcpy ( d_text, text, n * sizeof ( unsigned char ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_state_supply, state_supply, ( m * p_size + 1 ) * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_state_final, state_final, ( m * p_size + 1 ) * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );
	checkCudaErrors ( cudaMemcpy ( d_out, h_out, n * sizeof ( unsigned int ), cudaMemcpyHostToDevice ) );

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
	texture_memory_kernel<<dimGrid, dimBlock>>>( d_text, d_out, m, n, p_size, alphabet, numBlocks );
	checkCUDAError("kernel invocation");
	
	cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	
	cudaEventElapsedTime ( &time, start, stop );
	
	cudaEventDestroy ( start );
	cudaEventDestroy ( stop );

	//Get back the results from the device
	cudaMemcpy ( h_out, d_out, n * sizeof ( unsigned int ), cudaMemcpyDeviceToHost );
   
  	//Look at the results
  	int i, matches = 0;
  	for ( i = 0; i < n; i++ )
  	{
  		int count = 0;
  		if(h_out[i] == 0) continue;
  		for (int j = 0; j < D; ++j)
                {
                    if (h_out[i] & (1 << j))
                    {
                       // cout << "Word " << arr[j] << " appears from "
                        //    << i - arr[j].size() + 1 << " to " << i << endl;
                        count++;
                    }
                }
  		matches += count;
  	}
  	
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

	texture_memory_kernel_wrapper(M,charText,text.size(),D,26,goToTable,f,out);
	
	return 0;
}




