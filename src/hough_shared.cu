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
// Constant memory use for the cosine and sine tables
// Initialize in main and pass to device
__constant__ double d_Cos[degreeBins];
__constant__ double d_Sin[degreeBins];


//*****************************************************************
// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
// Now we are using constant memory for the cosine and sine values 
// to avoid recomputing them in the kernel
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, double rMax, double rScale)
{
  // Calculate the global thread ID
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  // Calculate local IDs
  int locID = threadIdx.x;

  // Check if the thread is out of bounds
  if (gloID >= w * h) return;      // in case of extra threads in block

  // Declare shared memory for the accumulator
  __shared__ int localAcc[rBins * degreeBins];
  
  // Initialize the shared memory to 0
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x){
    localAcc[i] = 0;
  }
  __syncthreads(); // Synchronize threads


  // Calculate the center of the image
  int xCent = w / 2;
  int yCent = h / 2;

  // Calculate the x and y coordinates of the pixel
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  // Check if the pixel value is greater than 0
  if (pic[gloID] > 0)
    {
      // Iterate over the degree bins
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
          // Use constant memory for the cosine and sine values
          double r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;
          // Use of atomicAdd to prevent any race conditions
          atomicAdd(&acc[rIdx * degreeBins + tIdx], 1);
        }
    }
    __syncthreads(); // Synchronize threads

  // Copy the shared memory to the global memory
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x){
    atomicAdd(&acc[i], localAcc[i]);
  }
}


// Function to get points from the accumulator
double calculateTransformedPoints(double value, char op, double angle) {
  // Check the operator and return the value
  if (op == '+') {
    // if the operator is +, return the value + 1000 * angle
    return value + 1000 * (angle);
  }
  else if (op == '-') {
    // if the operator is -, return the value - 1000 * angle
    return value - 1000 * (angle);
  }
  else {
    return 0.0; //default value if operator is not valid
  }
}


void draw_Detected_Lines(cv::Mat& color_image, int *h_hough, int w, int h, double rScale, double rMax, int threshold){
  // Create a vector to store the detected lines by the Hough Transform
  std::vector<std::pair<cv::Vec2f, int>> detected_lines;
  // Iterate over the accumulator
  for (int r = 0; r < rBins; r++) {
    for (int t = 0; t < degreeBins; t++) {
      int index = r * degreeBins + t;
      int weight = h_hough[index];

      // Check if the weight is greater than the threshold
      if (weight > threshold) {
        double rValue = (r * rScale) - rMax;
        double tValue = t * radInc;
        // Add the detected line to the vector
        detected_lines.push_back(std::make_pair(cv::Vec2f(tValue, rValue), weight));
      }
    }
  }

  // Sort the lines by weight in descending order
  std::sort(detected_lines.begin(), detected_lines.end(), [](const std::pair<cv::Vec2f, int>& a, const std::pair<cv::Vec2f, int>& b) {
    return a.second > b.second;
  });

  // Draw the detected lines on the input image
  for (int i = 0; i < detected_lines.size(); i++){
    cv::Vec2f line = detected_lines[i].first;
    double theta = line[0];
    double rho = line[1];
    // Get the sine and cosine values
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);

    // Get the x and y values
    double x_origin = (w / 2) + (rho * cosTheta);
    double y_origin = (h / 2) - (rho * sinTheta);
    // Get the transformed points for starting and ending points
    double x1 = calculateTransformedPoints(x_origin, '+', sinTheta);
    double x2 = calculateTransformedPoints(x_origin, '-', sinTheta);
    double y1 = calculateTransformedPoints(y_origin, '+', cosTheta);
    double y2 = calculateTransformedPoints(y_origin, '-', cosTheta);

    // Round the values
    int x_start = cvRound(x1);
    int y_start = cvRound(y1);
    int x_end = cvRound(x2);
    int y_end = cvRound(y2);

    // Draw the line on the image
    cv::line(color_image, cv::Point(x_start, y_start), cv::Point(x_end, y_end), cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
  }

  // Display the image with the detected lines
  cv::imwrite("src/output/output_constant.png", color_image);
  printf("SUCCES: Output image saved as 'output.png'\n");

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
  cudaMalloc ((void **) &d_Cos, sizeof(double) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof(double) * degreeBins);

  // CPU calculation --------------------------------------------------------------
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

  // GPU calculation --------------------------------------------------------------
 
  // copy the pre-computed values to the device constant memory
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(double) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(double) * degreeBins);

  // pre-compute values for the radius
  double rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  double rScale = 2 * rMax / rBins;
  
  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen
  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  // allocate memory in the device
  cudaMalloc ((void **) &d_in, sizeof(unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof(int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof(int) * degreeBins * rBins);

  // Create events for timing purposes
  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);
  
  // Record the Kernel execution time
  cudaEventRecord (start);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
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

  // compare CPU and GPU results --------------------------------------------------------------
  printf ("Comparing CPU and GPU results...\n");
  int mismatches = 0;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i]) 
    {
      printf (" -> Calculation mismatch at index: %i - CPU: %i | GPU: %i\n", i, cpuht[i], h_hough[i]);
      mismatches++;
    }
  }
  if (mismatches == 0)
    printf ("SUCCESS: No mismatches found\n\n");
  else
    printf ("ERROR: %d mismatches found\n\n", mismatches);


  // Image Generation --------------------------------------------------------------
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
  double threshold = mean + (2.5 * stddev);
  printf("Info -> Threshold value: %f\n", threshold);
  printf("Info -> Mean value: %f, Standard deviation: %f\n", mean, stddev);
  
  // Load the image into a cv::Mat object
  cv::Mat img(h, w, CV_8UC1, inImg.pixels);// Load PGM data into grayscale Mat
  // Convert grayscale image to BGR color image for red line drawing
  cv::Mat color_img;
  cv::cvtColor(img, color_img, cv::COLOR_GRAY2BGR);

  // Draw the detected lines on the image
  draw_Detected_Lines(color_img, h_hough, w, h, rScale, rMax, threshold);

  // Free memory --------------------------------------------------------------
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
