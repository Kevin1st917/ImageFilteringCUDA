
/**
* @file fft.h
* @author Melih Altun @2015-2018
**/

#include <string.h>
#include <stdlib.h>
//#include <stdbool.h>
#include <math.h>
#include <malloc.h>

#ifndef FFT_INCLUDE
#define FFT_INCLUDE

#if !defined (PI)
#define PI 3.141592653589793
#endif

#if !defined (lin_index)
#define lin_index(i, j, numCol)  ( ((i)*(numCol))+(j) )   //2D to 1D array
#endif

//#define MAGNITUDE
//#define PHASE

typedef double floating;
//floating can be assigned to float, double or long double depending on the required precission.

typedef struct _fft_instance{
	size_t dftSize;
	floating *Re;
	floating *Im;
#if defined (MAGNITUDE)
	floating *abs;
#endif
#if defined (PHASE)
	floating *angle;
#endif
}fft_instance;

typedef struct _fft2_instance{
	size_t rows;
	size_t cols;
	floating *Re;
	floating *Im;
#if defined (MAGNITUDE)
	floating *abs;
#endif
#if defined (PHASE)
	floating *angle;
#endif
}fft2_instance;

// Buffers used by Bluestein algorithm.
typedef struct bluestein_buffers_ {
	floating *wr;
	floating *wi;
	floating *yr;
	floating *yi;
	floating *fyr;
	floating *fyi;
	floating *vr;
	floating *vi;
	floating *fvr;
	floating *fvi;
	floating *gr;
	floating *gi;
	floating *fgr;
	floating *fgi;
}bluestein_buffers;

/* Prototypes for internal functions*/
//two fft algorithms are used in this code as internal functions: cooley_tuckey radix 2 and bluestein
void cooley_tuckey(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

void bluestein(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

void fft_core(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

bool compute_fftSize(size_t N, size_t *fftSize, bluestein_buffers *buffers);

int init_bluestein_buffers(bluestein_buffers *buffers, size_t N, size_t fftSize);

int clear_bluestein_buffers(bluestein_buffers *buffers);

bool power_of_2(size_t N);

void primeFactorizer(size_t N, int *factors);

void complexMultiplication(floating z1r, floating z1i, floating z2r, floating z2i, floating *yr, floating *yi);

void complexReciprocal(floating zr, floating zi, floating *yr, floating *yi);

void set_fft_instance(fft_instance *fft, size_t size);  //Initializer for fft instances
void set_fft2_instance(fft2_instance *fft, size_t N, size_t M);  //Initializer for 2D fft instances

void fft_real(fft_instance *inst, floating xr[], int clk, size_t N);  //fft for real inputs
void fft_complex(fft_instance *inst, floating xr[], floating xi[], int clk, size_t N);  //fft for complex inputs
void ifft_real(floating xr[], fft_instance *inst, int clk, size_t N);  //inverse fft with real output
void ifft_complex(floating xr[], floating xi[], fft_instance *inst, int clk, size_t N);  //inverse fft with complex output

void fft2_real (fft2_instance *inst, floating xr[], size_t N, size_t M);  //2D fft for real inputs
void fft2_complex(fft2_instance *inst, floating xr[], floating xi[], size_t N, size_t M);  //2D fft for complex inputs
void ifft2_real(floating xr[], fft2_instance *inst, size_t N, size_t M);  //inverse 2D fft with real output
void ifft2_complex(floating xr[], floating xi[], fft2_instance *inst, size_t N, size_t M);  //inverse 2D fft with complex output

void delete_fft_instance(fft_instance *fft);  //releases dynamic memory used by fft instaces
void delete_fft2_instance(fft2_instance *fft);  //releases dynamic memory used by 2D fft instaces

__device__ void dev_cooley_tuckey(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

__device__ void dev_bluestein(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

__device__ void dev_fft_core(floating Xr[], floating Xi[], floating xr[], floating xi[], bluestein_buffers *buffers, size_t N, size_t fftSize);

#endif
