// nvcc fft_cuda_2d.cu -lcublas -lcufft -arch=compute_52 -o fft_cuda_2d
//https://www.researchgate.net/figure/Computing-2D-FFT-of-size-NX-NY-using-CUDAs-cuFFT-library-49-FFT-fast-Fourier_fig3_324060154
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <cufft.h>
 
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

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




    float elapsedTime = 0;
    cufftHandle plan;
    cufftComplex *host_data = (cufftComplex*)malloc(NX*NY*sizeof(cufftComplex));
    cufftComplex *fft_data = (cufftComplex*)malloc(NX*NY*sizeof(cufftComplex));
    cufftComplex *dev_data;
    cudaEvent_t start,stop;
    
    //FEED INPUT
    srand(time(NULL));
    for(int i = 0;i<NX;i++){
        for(int j = 0;j<NY;j++){
            host_data[i*NY+j].x = (float)data[i*NY+j];  //rand()/(float)RAND_MAX;
            host_data[i*NY+j].y = 0.0;        
        }
    }

    //SHOW HOST DATA
    for(int i = 0;i<16;i++){
        printf("DATA: %3.1f %3.1f \n",host_data[i*NY+1].x,host_data[i*NY+1].y);
    }

    //ALLOCATE GPU MEMORY
    cudaMalloc((void**)&dev_data,sizeof(cufftComplex)*NX*NY);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    
    //COPY INPUT
    cudaMemcpy(dev_data,host_data,NX*NY*sizeof(cufftComplex),cudaMemcpyHostToDevice);
    
    //CREATE CUFFT PLAN
    cufftPlan2d(&plan,NX,NY,CUFFT_C2C);
    
    //PERFORM COMPUTATION(fft and ifft)
    cufftExecC2C(plan,dev_data,dev_data,CUFFT_FORWARD);
    //COPY BACK RESULTS
    cudaMemcpy(fft_data,dev_data,sizeof(cufftComplex)*NX*NY,cudaMemcpyDeviceToHost);
    ofstream outfile2;
    outfile2.open("fft_data.txt");
    // int data2[220*220] = {0};
    for(int i = 0;i<NX;i++){
        for(int j = 0;j<NY;j++){
            if(j == NY - 1){
                outfile2<<fft_data[i*NY+j].x<<endl;
            }else{
                outfile2<<fft_data[i*NY+j].x<<","; 
            }     
        }
    }
    outfile2.close(); 
    


    cufftExecC2C(plan,dev_data,dev_data,CUFFT_INVERSE);//https://stackoverflow.com/questions/46562575/how-to-cuda-ifft
    
    //COPY BACK RESULTS
    cudaMemcpy(host_data,dev_data,sizeof(cufftComplex)*NX*NY,cudaMemcpyDeviceToHost);
    
    //GET CALCULATION TIME
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    
    //SHOW RESULTS
    for(int i = 0;i<16;i++){
        printf("DATA: %3.1f %3.1f \n",host_data[i*NY+1].x/(NX*NY),host_data[i*NY+1].y/(NX*NY));
    }
    ofstream outfile;
    outfile.open("output_data.txt");
    // int data2[220*220] = {0};
    for(int i = 0;i<NX;i++){
        for(int j = 0;j<NY;j++){
            // data2[i*NY+j] = host_data[i*NY+3].x/(NX*NY)
            if(j == NY - 1){
                outfile<<host_data[i*NY+j].x/(NX*NY)<<endl;
            }else{
                outfile<<host_data[i*NY+j].x/(NX*NY)<<","; 
            }
              
        }
    }
    outfile.close();
    //FREEE MEMORY
    cufftDestroy(plan);
    cudaFree(dev_data);
    free(host_data);
    printf("elapsed time %f\n",elapsedTime);
    printf("CUFFT Calculation COMPLETED IN : % 5.3f ms \n",elapsedTime);  
}
