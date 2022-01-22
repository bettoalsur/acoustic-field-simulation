#include <iostream>
#include "nvidiabook/common/book.h"
#include "timer.h"
#include <math.h> //para operaciones matematicas
#include <fstream>
#include <time.h>


//Parametros de la solucion
#define NTHREADS (512) // numero de hilos por bloque


//Parametros del problema
#define c (343.00)  // velocidad de propagacion en el aire
#define f (40000.00)  //frecuencia de excitacion
#define lambda (c/f)  //longitud de onda
#define omega (2*3.1416*f)  //frecuenca angular


__global__ void campoAcustico(float *d_y ,float *d_z ,float *d_PR, float *d_PI ,float xdis ,float ydis ,float dA,float nel)
{
	int i = threadIdx.x + NTHREADS*blockIdx.x;

	if (i < nel*nel)
	{
		float R = sqrt( xdis*xdis + d_z[i]*d_z[i] + (d_y[i] - ydis)*(d_y[i] - ydis) );
		d_PR[i] += (dA/R)*cos(omega*R/c);
		d_PI[i] += (dA/R)*sin(omega*R/c);
	}
}

__global__ void resultante(float *d_PR, float *d_PI, float *d_PA, float nel)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if (i < nel*nel)
	{
		d_PA[i] = sqrt( d_PR[i]*d_PR[i] + d_PI[i]*d_PI[i] );
	}
}


int main()
{
	//Campo de muestra
	float ymuestra = 0.040;
	float zmuestra = 0.100;
	int pps = 12;
	int nel;
	if (zmuestra > ymuestra)
	{	nel = (pps/lambda)*zmuestra;	}
	else
	{	nel = (pps/lambda)*ymuestra;	}

	if (nel%2!=0)
	{
		nel+=1;
	}

	//Disco emisor
	float r = 0.025;
	int neldis = ((2*pps*r)/lambda);
	if (neldis%2!=0)
	{
		neldis+=1;
	}

	//Campo && Disco (vacios)
	size_t sizeCampo = nel*nel*sizeof(float);
	size_t sizeDisco = neldis*neldis*sizeof(float);

	float* y = (float*)malloc(sizeCampo);
	float* z = (float*)malloc(sizeCampo);
	float* xdis = (float*)malloc(sizeDisco);
	float* ydis = (float*)malloc(sizeDisco);

	//Llenar los vectores 
	for (int cont1 = 0; cont1 < nel ; cont1++)
	{
		for (int cont2 = 0; cont2 < nel ; cont2++)
		{
			y[cont1+nel*cont2]=cont2*(ymuestra/(nel-1)); //se llena saltando
			z[cont1*nel+cont2]=cont2*(zmuestra/(nel-1)); //se llena en orden
		}
	}

	float dxy = (2*r/(neldis-1));
	float dA = dxy*dxy;
	for (int cont1 = 0; cont1 < neldis ; cont1++)
	{
		for (int cont2 = 0; cont2 < neldis ; cont2++)
		{
			xdis[cont1+neldis*cont2]=-r+cont2*dxy; //se llena saltando
			ydis[cont1*neldis+cont2]=-r+cont2*dxy; //se llena en orden
		}
	}

	//Vector de Presion (vacio) HOST
	float* P = (float*)malloc(sizeCampo);

	
	////// Variables en la GPU //////

	float* d_y; cudaMalloc(&d_y,sizeCampo);		//ancho del campo
	float* d_z; cudaMalloc(&d_z,sizeCampo);		//largo del campo
	float* d_PR; cudaMalloc(&d_PR,sizeCampo);		//Presion REAL en el campo
	float* d_PI; cudaMalloc(&d_PI,sizeCampo);		//Presion IMAG en el campo
	float* d_PA; cudaMalloc(&d_PA,sizeCampo);		//Presion IMAG en el campo


	// Comienza medicion de TIEMPO
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);


	////// Copia de informacion del Host a la GPU //////

	cudaMemcpy(d_y,y,sizeCampo,cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,z,sizeCampo,cudaMemcpyHostToDevice);


	////// Resuleve en la GPU (Kernel) //////

	for (int rr = 0 ; rr < neldis*neldis ; rr++)
	{
		if ( sqrt(xdis[rr]*xdis[rr] + ydis[rr]*ydis[rr]) <= r )
		{
			campoAcustico<<<(nel*nel+(NTHREADS-1))/NTHREADS,NTHREADS>>>(d_y , d_z , d_PR , d_PI , xdis[rr] , ydis[rr] , dA, nel);
		}
	}

	resultante<<<(nel*nel+(NTHREADS-1))/NTHREADS,NTHREADS>>>(d_PR , d_PI, d_PA, nel);


	////// Copia los resultados de la GPU al HOST //////
	cudaMemcpy(P,d_PA,sizeCampo,cudaMemcpyDeviceToHost);


	// Termina de medir el TIEMPO
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float tempo;
	cudaEventElapsedTime(&tempo, start , stop);
	tempo/=1000;
	

	////// Libero la memoria de la GPU //////
	cudaFree(d_y);
	cudaFree(d_z);
	cudaFree(d_PR);
	cudaFree(d_PI);
	cudaFree(d_PA);

	// Mensajes para la consola
	std::cout << "Numero total de elementos: " << nel*nel + neldis*neldis << std::endl;
	std::cout << "Campo: " << nel*nel << " && Disco: " << neldis*neldis << std::endl;
	std::cout << "Tempo CUDA: " << tempo << " segundos" << std::endl;

	//Creacion de los archivos de texto

	std::ofstream file1;
	file1.open ("C:/Users/lemos/proyectoGC/datos.txt");
	for (int contxx = 0 ; contxx < nel*nel ; contxx++)
	{	file1 << y[contxx] << "	" << z[contxx] << "	" << P[contxx] << "\n";  }
	file1.close();

	std::ofstream file2;
	file2.open ("C:/Users/lemos/proyectoGC/tiempo.txt");
	file2 << tempo; 
	file2.close();

	std::ofstream file3;
	file3.open ("C:/Users/lemos/proyectoGC/parametros.txt");
	file3 << c << "	" << f << "	" << ymuestra << "	" << zmuestra << "	" << pps << "	" << r; 
	file3.close();

	///////////////////////////////
	////// Calculo en la CPU //////

	float RR;
	float* Preal = (float*)malloc(sizeCampo);
	float* Pimag = (float*)malloc(sizeCampo);
	float* Pabs = (float*)malloc(sizeCampo);

	double cputime = getTimeStamp();		// Comienza contar tiempo

	for (int rr = 0 ; rr < neldis*neldis ; rr++)
	{
		if ( sqrt(xdis[rr]*xdis[rr] + ydis[rr]*ydis[rr]) <= r )
		{
			for (int ii = 0 ; ii < nel*nel ; ii++)
			{
				RR = sqrt( xdis[rr]*xdis[rr] + z[ii]*z[ii] + (y[ii] - ydis[rr])*(y[ii] - ydis[rr]) );
				Preal[ii] += (dA/RR)*cos(omega*RR/c);
				Pimag[ii] += (dA/RR)*sin(omega*RR/c);
			}
		}
	}

	for (int ii = 0 ; ii < nel*nel ; ii++)
	{
		Pabs[ii] = sqrt( Preal[ii]*Preal[ii] + Pimag[ii]*Pimag[ii] );
	}

	cputime = getTimeStamp() - cputime;		// Termina de contar tiempo
	cputime/=1000;

	std::cout << "Tempo CPU: " << cputime << " segundos" << std::endl;

	//Creacion de los archivos de texto

	std::ofstream file4;
	file4.open ("C:/Users/lemos/proyectoGC/datosCPU.txt");
	for (int contxx = 0 ; contxx < nel*nel ; contxx++)
	{	file4 << y[contxx] << "	" << z[contxx] << "	" << Pabs[contxx] << "\n";  }
	file4.close();

	std::ofstream file5;
	file5.open ("C:/Users/lemos/proyectoGC/tiempoCPU.txt");
	file5 << cputime; 
	file5.close();

	std::cin.get();
	return 0;
}