#include <iostream>



using namespace std;

void kernel(int n,int m,int blockIdxx,int threadIdxx,int offset,int blockSize,
																int threadsPerBlock){

    int i,idx = threadIdxx;

    for(i=0;i<offset;i++){
    	int sharedIndex = idx + i*threadsPerBlock;
    	int globalIndex = sharedIndex+blockIdxx*blockSize;

    	cout<<sharedIndex<<"   "<<globalIndex<<endl;
    }
}

int main(){
	int n = 5000;//size of total string
	int m = 50;//size of patlen
	int SHAREDMEMPERBLOCK = 500;
	int threadsPerBlock = 5;



	int sm_size = SHAREDMEMPERBLOCK;//devProp.sharedMemPerBlock/2; //so that atleast 2 blocks can be scheduled simultaneously
    
    int conceptualBlockSize = SHAREDMEMPERBLOCK - m +1;

    int n_blocks = (n-1)/(conceptualBlockSize) + 1;//number of blocks
    int offset = sm_size/threadsPerBlock;// number of characters each thread loads into shared mem =D
   	

	
	int blockIdxx=n_blocks-1 ;
	int threadIdxx=4 ;
	//kernel
	kernel(n,m,blockIdxx,threadIdxx,offset,conceptualBlockSize,threadsPerBlock);
	cout<<endl<<conceptualBlockSize<<endl;
	return 0;
}