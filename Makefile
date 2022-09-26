# Makefile
 
# *****************************************************
# Target 
TARGET = main
# Compiler
CC = g++-12

# compiler flags:
  #  -g         - adds debugging info to the executable file
  #  -Wall      - turn on most compiler warnings
  # -std=c++17  - turn on C++17 standard
  # -I          - include path ("./" for current path)
  # -fopenmp    - OpenMP: parallelisation
  # -03         - turn ON level O3 compiler optimization
 
CFLAGS = -Wall -g -std=c++17 -I ./ -fopenmp -O3 -Wno-sign-compare
 
# ****************************************************
 
main: $(TARGET).o
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).o

$(TARGET).o: $(TARGET).cpp
	$(CC) $(CFLAGS) -c $(TARGET).cpp

clean:
	rm -f $(TARGET) *.o