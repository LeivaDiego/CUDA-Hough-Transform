# Paths to OpenCV in WSL
OPENCV_INCLUDE_PATH = /usr/include/opencv4
OPENCV_LIB_PATH = /usr/lib/x86_64-linux-gnu

# OpenCV Libraries
OPENCV_LIBS = `pkg-config --cflags --libs opencv4`

# Warnings to suppress
SUPPRESS_WARNINGS = -diag-suppress=611

# Linker flags
all: pgm.o	houghBase houghGlobal


# Compiling the code for the base version
houghBase:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o houghBase

# Compiling the code for the global memory version with the pgm.o file
houghGlobal:	hough_global.cu pgm.o
	nvcc $(SUPPRESS_WARNINGS) hough_global.cu pgm.o -o houghGlobal -I$(OPENCV_INCLUDE_PATH) -L$(OPENCV_LIB_PATH) $(OPENCV_LIBS)

# Compiling the code for the pgm file handling
pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
