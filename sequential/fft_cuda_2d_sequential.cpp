// nvcc fft_cuda_2d.cu -lcublas -lcufft -arch=compute_52 -o fft_cuda_2d
//https://www.researchgate.net/figure/Computing-2D-FFT-of-size-NX-NY-using-CUDAs-cuFFT-library-49-FFT-fast-Fourier_fig3_324060154

#include "fft.h" 
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define DIM 4*65536//65536 = 256 * 256
#define NX 220
#define NY 220
using namespace std;



int main()
{
    int n = 0; //n is the number of the integers in the file ==> 12
    int data[220*220];
    int x;

    ifstream File;
    File.open("lenna_grayscale.txt");
    if(!File.is_open()){
        cout<<"It failed"<<endl;
        return 0;
    }

    while(File>>x){
        data[n] = x; 
        n++;
    }

    File.close();
    cout<<"n : "<<n<<endl;
    for(int i=0;i<n;i++){
        cout << data[i] << " ";
    }

    double buff[NX*NY];
    for(int i = 0; i < NX*NY; i++){
	    buff[i] = (double)data[i];
	}

	clock_t start_time, elapsed_time;
	double elapsed;

    floating Dr[NX * NY], Di[NX * NY], ddr[NX * NY], ddi[NX * NY], ddm[NX * NY];

    fft2_instance fft2;

	set_fft2_instance(&fft2, NX, NY);

	start_time = clock();
	
	fft2_real(&fft2, buff, NX, NY);

	elapsed_time = clock() - start_time;
	elapsed = elapsed_time / ((double)CLOCKS_PER_SEC);
	printf("Time to process 2D real FFT (N = 53, M = 53): %f seconds\n\n", elapsed);

	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			Dr[lin_index(i, j, NX)] = fft2.Re[lin_index(i, j, NX)];
			Di[lin_index(i, j, NY)] = fft2.Im[lin_index(i, j, NY)];
		}
	}

    // ofstream outfile2;
    // outfile2.open("fft_data.txt");

    // for(int i = 0;i<NX;i++){
        // for(int j = 0;j<NY;j++){
            // if(j == NY - 1){
                // outfile2<<[i*NY+j].x<<endl;
            // }else{
                // outfile2<<[i*NY+j].x<<","; 
            // }     
        // }
    // }
    // outfile2.close(); 

	start_time = clock();

	ifft2_real(ddm, &fft2, NX, NY);

	elapsed_time = clock() - start_time;
	elapsed = elapsed_time / ((double)CLOCKS_PER_SEC);
	printf("Time to process 2D real iFFT (N = 53, M = 53): %f seconds\n\n", elapsed);


    ofstream outfile;
    outfile.open("output_data.txt");
    // int data2[220*220] = {0};
    for(int i = 0;i<NX;i++){
        for(int j = 0;j<NY;j++){
            // data2[i*NY+j] = host_data[i*NY+3].x/(NX*NY)
            if(j == NY - 1){
                outfile<<ddm[i*NY+j]<<endl;
            }else{
                outfile<<ddm[i*NY+j]<<","; 
            }
              
        }
    }
    outfile.close();

    delete_fft2_instance(&fft2);





}
