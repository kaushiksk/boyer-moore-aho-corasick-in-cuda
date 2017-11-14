#include<bits/stdc++.h>
using namespace std;
#include<ctime>


int main(){
	
	int a[100];
	unsigned int * b = (unsigned int*)a;

	string P="ATGC";
	double n;
	cin>>n;
	
	ofstream output_file("data.txt");

	srand(time(NULL));
	for(long long int i=0; i<(int) 1024*1024*n; i++)
		output_file<<P[rand() % 4];
	

}