# Makefile
 
# *****************************************************
# Compiler
CC = g++-12

# compiler flags:
  #  -g         - adds debugging info to the executable file
  #  -Wall      - turn on most compiler warnings
  # -std=c++17  - turn on C++17 standard
  # -isysroot   - this is added for fixing include problems with macOS Catalina
CFLAGS = -Wall -g -std=c++17 -I ./ -fopenmp -O3 -Wno-sign-compare -DEIGEN_STACK_ALLOCATION_LIMIT=0
 
# ****************************************************
 
main: main.o matrix.o vector.o
	$(CC) $(CFLAGS) -o main main.o matrix.o vector.o

main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp
 
matrix.o: matrix.cpp
	$(CC) $(CFLAGS) -c matrix.cpp

vector.o: vector.cpp
	$(CC) $(CFLAGS) -c vector.cpp
clean:
	rm -f main *.o