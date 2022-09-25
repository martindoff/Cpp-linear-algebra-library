# Cpp-linear-algebra-library

## Description

A lightweight linear algebra library to perform basic matrix / vector operations using a simple syntax.

The `Vector` and `Matrix` template classes are based on the `std::valarray`container for numerical efficiency. 

The library can be used for rapid prototyping of C++ projects involving matrix / vector operations and linear algebra when performance is not a key requirement. It can easily be integrated and used intuitively. 

## Install

In a terminal, once in your chosen directory: 
```
git clone https://github.com/martindoff/Cpp-linear-algebra-library.git
```
Then: 

```
cd Cpp-linear-algebra-library
```

## Build


### Makefile
You will first need to set correctly your compiler in the Makefile (if you choose to compile using GNU make). For example, I personally use g++ version 12 for this project, hence:
```
CC = g++-12 
```
But `CC = g++` would call your default compiler associated with g++.

Once set:
```
make
```
To run the example program:
```
./main
```

### Command line
For example, with version 12 of g++, using the C++17 standard, setting compiler optimisation level 03, using OpenMP, and including the library to main: 

```
g++-12 -Wall -g -std=c++17 -I ./ -fopenmp -O3 main.cpp matrix.cpp vector.cpp -o main
```
### Integration in your project
The only files that are required are: `vector.h`, `matrix.h`, `vector.cpp`, `matrix.cpp`, so you can just copy paste them and link them to your project.

## Demo program

The `main.cpp` file contains a demo program that shows usage for most operations offered by this library. It can be used as a tutorial to use the library. For example, to create a random matrix of `double` of size 2-by-3: 

```c++
Matrix<double> A = rand(2, 3); 
```
Multiply the matrix by `2.0`, add `1.0` and store result in a new matrix:

```c++
Matrix<double> B = 2.0*A + 1.0; 
```
Add both matrices and print result:
```c++
#include <iostream>
  ...
std::cout << A + B; 
```
Matrix multiplication: 

```c++
Matrix<double> C = A * B.t(); 
```
Inversion: 
```c++
Matrix<double> D = C.inv(); 
```
## Testing
Although some basic testing of all operations was carried out "manually" in the `main.cpp` file, this is not sufficient to garantee the library reliability. Extensive testing was performed for a select number of key operations that cover most of the library. These are:

* matrix-vector multiplication
* vector-vector addition
* matrix inversion

The tests can be found in the directory `test/` and consist in a comparison with the Eigen library of the result of these operations on objects of various dimensions (from 2 to 500). For a given dimension, 100,000 experiments involving randomly generated objects were carried out and the success rate was measured against a given tolerance for the infinity norm of the difference of both results. The success rate was shown to be >99.9% with a tolerance of 1e-6 for all tested operations.

The library will also be updated and tested more in depth as I use it in my projects. 

## Benchmarking

I was curious to assess the performances of my library against some state-of-the-art linear algebra libraries. Most of them use sophisticated optimisation of the memory architetcture in order to minimise cache misses, e.g. by block factoring the input matrices before performing an operation. Although my goal was not to compete with these libraries, and despite my implementation is quite rudimentary in comparison, I was surprised to see that the performances of my custom library are competitive for low dimensions (matrix sizes below 15) and not too terrible for higher dimensions - staying within an order of magnitude from the state of the art in terms of average CPU time. Of course this is probably insuficient for performance critical applications, but not so bad for rapid developments.

To assess the performances of the present library, the key operations aforementioned in the testing section were run using various linear algebra libraries over a range of problem dimensions. For a given operation, library and problem dimension, 100,000 executions of the operation were conducted on random objects and the average time to completion was recorded. The results are shown in the Figure below for low dimensions and for the matrix inversion operation:

As can be seen from the Figure, the custom library is quite competitive in that range. However, state-of-the-art libraries are taking over at higher dimensions:

The custom library is about 3 times slower on average than the fastest linear algebra library tested for high dimensions. 

## Future work
Future developments could involve improving the library to make it run faster on high dimensional problems or extend its capabilities. For example:
* replacing the container `std::valarray` by plain C-style arrays. I had initially designed the library relying on the `std::vector` container but could divide the average time to completion by about 3 just by replacing the later container by `std::valarray` containers that are more optimised for numerical computations. I would be curious to see what would be the gain (or pain) resulting in resorting to pure arrays for the data members of the `Matrix` and `Vector` classes. 
* Block decomposition of the input matrices before performing an operation can speed up computations by minimising cache misses. This is an interesting avenue of development for improving the performances of the library.
* Much more linear algebra operations could be added such as a QR decomposition to solve, e.g. least square problems. 
