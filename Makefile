# Paths to OpenCV in WSL
OPENCV_INCLUDE_PATH = /usr/include/opencv4
OPENCV_LIB_PATH = /usr/lib/x86_64-linux-gnu

# OpenCV Libraries
OPENCV_LIBS = `pkg-config --cflags --libs opencv4`

# Warnings to suppress
SUPPRESS_WARNINGS = -diag-suppress=611

# Directory paths
SRC_DIR = src
INPUT_DIR = $(SRC_DIR)/input
OUTPUT_DIR = $(SRC_DIR)/output
COMMON_DIR = $(SRC_DIR)/common
EXEC_DIR = $(SRC_DIR)/executables

# Executables
all: pgm.o houghBase houghGlobal houghConstant

# Compiling the base version
houghBase: $(SRC_DIR)/houghBase.cu $(COMMON_DIR)/pgm.o
	nvcc $(SRC_DIR)/houghBase.cu $(COMMON_DIR)/pgm.o -o $(EXEC_DIR)/houghBase

# Compiling the global memory version with OpenCV flags
houghGlobal: $(SRC_DIR)/hough_global.cu $(COMMON_DIR)/pgm.o
	nvcc $(SUPPRESS_WARNINGS) $(SRC_DIR)/hough_global.cu $(COMMON_DIR)/pgm.o -o $(EXEC_DIR)/houghGlobal -I$(OPENCV_INCLUDE_PATH) -L$(OPENCV_LIB_PATH) $(OPENCV_LIBS)

# Compiling the constant memory version with OpenCV flags
houghConstant: $(SRC_DIR)/hough_constant.cu $(COMMON_DIR)/pgm.o
	nvcc $(SUPPRESS_WARNINGS) $(SRC_DIR)/hough_constant.cu $(COMMON_DIR)/pgm.o -o $(EXEC_DIR)/houghConstant -I$(OPENCV_INCLUDE_PATH) -L$(OPENCV_LIB_PATH) $(OPENCV_LIBS)

# Compiling the pgm file handling
pgm.o: $(COMMON_DIR)/pgm.cpp
	g++ -c $(COMMON_DIR)/pgm.cpp -o $(COMMON_DIR)/pgm.o
