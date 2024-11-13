all: pgm.o	houghBase houghGlobal


houghBase:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o houghBase

houghGlobal:	hough_global.cu pgm.o
	nvcc hough_global.cu pgm.o -o houghGlobal

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
