//Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>

// Constants
const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const double radInc = degreeInc * M_PI / 180;

//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  double rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  double rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            double theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                double r = xCoord * cos(theta) + yCoord * sin(theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// TODO usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
//__constant__ float d_Cos[degreeBins];
//__constant__ float d_Sin[degreeBins];

//*****************************************************************
//TODO Kernel memoria compartida
// __global__ void GPU_HoughTranShared(...)
// {
//   //TODO
// }
//TODO Kernel memoria Constante
// __global__ void GPU_HoughTranConst(...)
// {
//   //TODO
// }

// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, double rMax, double rScale, double *d_Cos, double *d_Sin)
{
  // Calculate the global thread ID
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID >= w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  //TODO explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  //TODO eventualmente usar memoria compartida para el acumulador

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          //TODO utilizar memoria constante para senos y cosenos
          //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //Linea probada y provoca un aumento de 2X del tiempo 
          // - Ademas induce discrepancias en la comparacion de valores con CPU 8k+
          double r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          //debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
          atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
        }
    }

  //TODO eventualmente cuando se tenga memoria compartida, copiar del local al global
  //utilizar operaciones atomicas para seguridad
  //faltara sincronizar los hilos del bloque en algunos lados

}

//*****************************************************************
int main (int argc, char **argv)
{
  // Check for proper usage and input file
  if (argc != 2)
  {
    printf("ERROR -> Invalid number of arguments | Usage: %s <image.pgm>\n", argv[0]);
    exit(1);
  }

  // Check for proper file format (PGM)
  if (strstr (argv[1], ".pgm") == NULL)
  {
    printf ("ERROR -> Invalid file format. Please use a .pgm file\n");
    exit(1);
  }

  // Check for file existence
  FILE *file = fopen (argv[1], "r");
  if (file == NULL)
  {
    printf ("ERROR -> Input File '%s' not found.\n", argv[1]);
    exit(1);
  }
  fclose (file); // Close file after checking


  // Variables
  int i;

  // Load the PGM image after successful checks
  PGMImage inImg (argv[1]);

  // Create the accummulator in the host
  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // Allocate memory for the cosine and sine values
  double* d_Cos;
  double* d_Sin;

  // Allocate memory for the cosine and sine values
  cudaMalloc ((void **) &d_Cos, sizeof(double) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof(double) * degreeBins);

  // CPU calculation
  // Record the CPU execution time
  auto cpu_start = std::chrono::high_resolution_clock::now();
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_elapsed = cpu_end - cpu_start;
  printf("CPU Execution Time: %f ms\n", cpu_elapsed.count());

  // pre-compute values to be stored
  double *pcCos = (double *) malloc (sizeof(double) * degreeBins);
  double *pcSin = (double *) malloc (sizeof(double) * degreeBins);
  double rad = 0;
  // fill the arrays with the pre-computed values
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  // pre-compute values for the radius
  double rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  double rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  // copy the pre-computed values to the device
  cudaMemcpy(d_Cos, pcCos, sizeof(double) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof(double) * degreeBins, cudaMemcpyHostToDevice);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen
  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  // allocate memory in the device
  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // Create events for timing purposes
  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  
  // Record the Kernel execution time
  int blockNum = ceil (w * h / 256);
  cudaEventRecord (start);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  cudaEventRecord (stop);

  // Wait for the stop event and synchronize threads
  cudaEventSynchronize (stop);
  float elapsedTime = 0;
  cudaEventElapsedTime (&elapsedTime, start, stop);
  printf ("GPU Execution Time for Global: %f ms\n\n", elapsedTime);

  // Destroy the events
  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // compare CPU and GPU results
  printf ("Comparing CPU and GPU results...\n");
  int mismatches = 0;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i]) 
    {
      printf (" -> Calculation mismatch at %i - CPU: %i | GPU: %i\n", i, cpuht[i], h_hough[i]);
      mismatches++;
    }
  }
  if (mismatches == 0)
    printf ("SUCCESS: No mismatches found\n\n");
  else
    printf ("ERROR: %d mismatches found\n\n", mismatches);


  // Image Generation
  printf("Generating output image...\n");
  // Initialize mean and standard deviation for thresholding
  double mean = 0, stddev = 0;
  int total_elements = degreeBins * rBins;
  
  // Calculate the mean and standard deviation for thresholding
  for (int i = 0; i < total_elements; i++) 
  {
    mean += h_hough[i];
  }
  mean /= total_elements;

  for (int i = 0; i < total_elements; i++) 
  {
    stddev += pow(h_hough[i] - mean, 2);
  }
  stddev = sqrt(stddev / total_elements);

  // Threshold value is set to 2 standard deviations above the mean
  double threshold = mean + 3 * stddev;
  // Line length to improve visualization
  int line_length = 1000;
  printf("Info -> Threshold value: %f\n", threshold);
  printf("Info -> Mean value: %f, Standard deviation: %f\n", mean, stddev);
  
  // Convert PGMImage to OpenCV Mat for line drawing and saving
  cv::Mat img(h, w, CV_8UC1, inImg.pixels); // Load PGM data into grayscale Mat
  cv::Mat color_img;
  cv::cvtColor(img, color_img, cv::COLOR_GRAY2BGR);
  
  // Draw lines on the image
  for (int rIdx = 0; rIdx < rBins; rIdx++) {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++) {
          if (h_hough[rIdx * degreeBins + tIdx] > threshold) {
              double theta = tIdx * radInc;
              double r = (rIdx * rScale) - rMax;
              double cosT = cos(theta), sinT = sin(theta);
              cv::Point pt1, pt2;
              pt1.x = cvRound(r * cosT + line_length * (-sinT));
              pt1.y = cvRound(r * sinT + line_length * cosT);
              pt2.x = cvRound(r * cosT - line_length * (-sinT));
              pt2.y = cvRound(r * sinT - line_length * cosT);
              cv::line(color_img, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
          }
      }
  }
  // Save the output image
  cv::imwrite("src/output/global_output.png", color_img);
  printf("SUCCES: Output image saved as 'output.png'\n");

  // free memory
  free(pcCos);
  free(pcSin);
  free(h_hough);
  free(cpuht);
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  cudaFree(d_in);
  cudaFree(d_hough);

  return 0;
}