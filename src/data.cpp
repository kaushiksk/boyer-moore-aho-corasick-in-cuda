#include<bits/stdc++.h>
using namespace std;
#include<ctime>


int main(){
	
	int a[100];
	unsigned int * b = (unsigned int*)a;

	string P="ATGC";
	double n;
	cin>>n;
	srand(time(NULL));
	for(long long int i=0; i<(int) 1024*1024*n; i++)
		cout<<P[rand() % 4];
	

}