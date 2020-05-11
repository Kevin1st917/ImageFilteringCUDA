// nvcc fft_cuda_2d_parallel.cu fft_cu -arch=compute_52 -o parallel

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include "fft.h" 
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#define TILE_DIM 16
#define DIM 4*65536//65536 = 256 * 256
#define NX 220
#define NY 220
using namespace std;

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void ColWiseFFT(floating* colVec_r, floating* row_fft_img_r, 
                           floating* colVec_i, floating* row_fft_img_i,
						   fft_instance col_fft, bluestein_buffers* buffers,
						   size_t fftSize, fft2_instance* fft2) {
    
	// Identify the row and column of the d_P element to work on
    int Row = blockIdx.x*TILE_DIM + threadIdx.x;
    int Col = blockIdx.y*TILE_DIM + threadIdx.y;

	if (Col < (int)NY) {
		if (Row < (int)NX) {
			colVec_r[Row] = row_fft_img_r[lin_index(Row, Col, NY)];  //a pointer can be used later instead of indexing
			colVec_i[Row] = row_fft_img_i[lin_index(Row, Col, NY)];  //speed up will not be significant though.
		}
		dev_fft_core(col_fft.Re, col_fft.Im, colVec_r, colVec_i, buffers, NX, fftSize);

		if (Row < (int)NX) {
			fft2->Re[lin_index(Row, Col, NY)] = col_fft.Re[Row];
			fft2->Im[lin_index(Row, Col, NY)] = col_fft.Im[Row];

		}
	}
}

int main()
{
	/* Starting data file read */
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

    // print dimention and data
    // cout<<"n : "<<n<<endl;
    // for(int i=0;i<n;i++){
        // cout << data[i] << " ";
    // }

    double buff[NX*NY];
    for(int i = 0; i < NX*NY; i++){
	    buff[i] = (double)data[i];
	}
	/* Data file read ended */

    floating Dr[NX * NY], Di[NX * NY], ddr[NX * NY], ddi[NX * NY], ddm[NX * NY];

    fft2_instance fft2;

	set_fft2_instance(&fft2, NX, NY);

	
	/* Starting fft for real inputs */
	// fft2_real(&fft2, buff, NX, NY);

	dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
	dim3 dimGrid;
	dimGrid.x = (NX + dimBlock.x - 1)/dimBlock.x;
	// dimGrid.y = (NY + dimBlock.y - 1)/dimBlock.y;
	dimGrid.y = 1;

	floating *xi;
	xi = (floating*)calloc(NX*NY, sizeof(floating));
	fft_instance row_fft, col_fft;
	floating *rowVec_r = NULL, *colVec_r = NULL, *rowVec_i = NULL, *colVec_i = NULL, *row_fft_img_r = NULL, *row_fft_img_i = NULL;
	bluestein_buffers buffers;
	bool powerOf2;
	size_t fftSize;
	register int i, j;

	if (NX < 1 || NY < 1)  //this can lead to a crash
		return 0;

	set_fft_instance(&row_fft, NY);
	set_fft_instance(&col_fft, NX);

	rowVec_r = (floating*)malloc(NY * sizeof(floating));
	colVec_r = (floating*)malloc(NX * sizeof(floating));
	rowVec_i = (floating*)malloc(NY * sizeof(floating));
	colVec_i = (floating*)malloc(NX * sizeof(floating));

	row_fft_img_r = (floating*)malloc(NX * NY * sizeof(floating));
	row_fft_img_i = (floating*)malloc(NX * NY * sizeof(floating));

	//Calculate 2D FFT by row column decomposition
	powerOf2 = compute_fftSize(NY, &fftSize, &buffers);

	for (i = 0; i < (int)NX; i++) {
		memcpy(rowVec_r, &buff[i*NY], NY*sizeof(floating));
		memcpy(rowVec_i, &xi[i*NY], NY*sizeof(floating)); //row-wise for loops replaced by memcpy

		fft_core(row_fft.Re, row_fft.Im, rowVec_r, rowVec_i, &buffers, NY, fftSize);

		memcpy(&row_fft_img_r[i*NY], row_fft.Re, NY*sizeof(floating));
		memcpy(&row_fft_img_i[i*NY], row_fft.Im, NY*sizeof(floating));
	}
	if (!powerOf2)
		clear_bluestein_buffers(&buffers);

	//row-wise FFT done. column-wise FFT is next
	powerOf2 = compute_fftSize(NX, &fftSize, &buffers);

    floating *dev_colVec_r, *dev_row_fft_img_r, *dev_colVec_i, *dev_row_fft_img_i;
	bluestein_buffers dev_buffers;
	fft2_instance dev_fft2;
	set_fft2_instance(&dev_fft2, NX, NY);

	cudaMalloc((void **)&dev_colVec_r, NX*sizeof(floating));
	cudaMalloc((void **)&dev_row_fft_img_r, NX*NY*sizeof(floating));
	cudaMalloc((void **)&dev_colVec_i, NX*sizeof(floating));
	cudaMalloc((void **)&dev_row_fft_img_i, NX*NY*sizeof(floating));
	cudaMalloc((void **)&dev_buffers, sizeof(bluestein_buffers));
	cudaMalloc((void **)&dev_fft2, NX*NY*sizeof(fft2_instance));

	cudaMemcpy(dev_colVec_r, colVec_r, NX*sizeof(floating), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_row_fft_img_r, row_fft_img_r, NX*NY*sizeof(floating), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_colVec_i, colVec_i, NX*sizeof(floating), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_row_fft_img_i, row_fft_img_i, NX*NY*sizeof(floating), cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_buffers, &buffers, sizeof(bluestein_buffers), cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_fft2, &fft2, NX*NY*sizeof(fft2_instance), cudaMemcpyHostToDevice);

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    checkCuda(cudaEventCreate(&start));

    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&stop));

    // Record the start event
    checkCuda(cudaEventRecord(start, NULL));
 
   // Execute the kernel
	ColWiseFFT<<<dimGrid , dimBlock>>>(dev_colVec_r, dev_row_fft_img_r, dev_colVec_i, dev_row_fft_img_i, col_fft, &dev_buffers, fftSize, &dev_fft2);

    // cudaDeviceSynchronize();

	// Record the stop event
    checkCuda(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCuda(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCuda(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("ElapsedTime = %f", msecTotal);
	
	// for (j = 0; j < (int)NY; j++) {
		// for (i = 0; i < (int)NX; i++) {
			// colVec_r[i] = row_fft_img_r[lin_index(i, j, NY)];  //a pointer can be used later instead of indexing
			// colVec_i[i] = row_fft_img_i[lin_index(i, j, NY)];  //speed up will not be significant though.
		// }
		// fft_core(col_fft.Re, col_fft.Im, colVec_r, colVec_i, &buffers, NX, fftSize);

		// for (i = 0; i < (int)NX; i++) {
			// fft2->Re[lin_index(i, j, NY)] = col_fft.Re[i];
			// fft2->Im[lin_index(i, j, NY)] = col_fft.Im[i];

		// }
	// }

	cudaMemcpy(colVec_r, dev_colVec_r, NX*sizeof(floating), cudaMemcpyDeviceToHost);
	cudaMemcpy(row_fft_img_r, dev_row_fft_img_r, NX*NY*sizeof(floating), cudaMemcpyDeviceToHost);
	cudaMemcpy(colVec_i, dev_colVec_i, NX*sizeof(floating), cudaMemcpyDeviceToHost);
	cudaMemcpy(row_fft_img_i, dev_row_fft_img_i, NX*NY*sizeof(floating), cudaMemcpyDeviceToHost);
	cudaMemcpy(&buffers, &dev_buffers, sizeof(bluestein_buffers), cudaMemcpyDeviceToHost);
	cudaMemcpy(&fft2, &dev_fft2, NX*NY*sizeof(fft2_instance), cudaMemcpyDeviceToHost);


	if (!powerOf2)
		clear_bluestein_buffers(&buffers);

	//2D FFT complete at this point

	//clean up memory
	delete_fft_instance(&row_fft);
	delete_fft_instance(&col_fft);

    cudaFree(dev_colVec_r);
    cudaFree(dev_row_fft_img_r);
	cudaFree(dev_colVec_i);
	cudaFree(dev_row_fft_img_i);
	cudaFree(&dev_buffers);
	cudaFree(&dev_fft2);

    free(xi);
	free(rowVec_r);
	free(rowVec_i);
	free(colVec_r);
	free(colVec_i);
	free(row_fft_img_r);
	free(row_fft_img_i);

	/* fft for real inputs ended */


	for (int i = 0; i < NX; i++) {
		for (int j = 0; j < NY; j++) {
			Dr[lin_index(i, j, NX)] = fft2.Re[lin_index(i, j, NX)];
			Di[lin_index(i, j, NY)] = fft2.Im[lin_index(i, j, NY)];
			// printf("%f \n", *fft2.Re);
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

	ifft2_real(ddm, &fft2, NX, NY);

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
    delete_fft2_instance(&dev_fft2);
}
