# Cpp-linear-algebra-library

## Description

A lightweight linear algebra library to perform basic matrix / vector operations using a simple syntax.

The `Vector` and `Matrix` template classes are based on the `std::valarray`container for numerical efficiency. 

The library can be used for rapid prototyping of C++ projects involving matrix / vector operations and linear algebra when performance is not a key requirement. It can easily be integrated in most projects and used intuitively. 

## Built with

* C++
* g++ compiler 
* (optional) GNU make

## Getting started

To integrate the library in your project, all you need to do is 

* download the `matrix.h` file and place it in the include path of your project
* prepend your main file (or any file in which you wish to use the library) with `#include "matrix.h"`

and, voilà! 

## Demo program


### Download
Clone the repository: 
```
git clone https://github.com/martindoff/Cpp-linear-algebra-library.git
```
Then: 

```
cd Cpp-linear-algebra-library
```

### Build


#### Build with a Makefile
You will first need to set correctly your compiler in the Makefile (if you choose to compile using GNU make). For example, I personally use g++ version 12 for this project, hence:
```
CC = g++-12 
```
But `CC = g++` would call your default compiler associated with g++.

Once set, build:
```
make
```
To run the example program:
```
./main
```

#### Build in command line
For example, with version 12 of g++, using the C++17 standard, setting compiler optimisation level 03, using OpenMP, and including the library to main: 

```
g++-12 -Wall -g -std=c++17 -I ./ -fopenmp -O3 main.cpp matrix.cpp vector.cpp -o main
```

### Usage

The `main.cpp` file contains a demo program that shows the usage for most operations offered by this library. It can serve as a tutorial to use the library. For example, to create a random matrix of `double` of size `2-by-3`: 

```c++
Matrix<double> A = rand(2, 3); 
```
Multiply the matrix by `2.0`, add `1.0` and store result in a new matrix:

```c++
Matrix<double> B = 2.0*A + 1.0; 
```
Add both matrices and print result:
```c++
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
Although some basic testing of all operations was carried out "manually" in the `main.cpp` file, this is not sufficient to garantee that the library functions reliably. Extensive testing was performed for key operations, covering most of the library. These are:

* matrix-vector multiplication
* vector-vector addition
* matrix inversion

The tests can be found in the `test/` directory  and consist in a comparison with the Eigen library of the result of these operations on objects of various dimensions. For a given dimension, 100,000 experiments involving randomly generated objects were carried out and the success rate was measured against a given tolerance for the infinity norm of the difference of both results. The success rate was shown to be >99.9% with a tolerance of 1e-6 for all tested operations.

The library will also be updated and tested as I use it in my projects. 

## Benchmarking

I was curious to assess the performances of my library against state-of-the-art linear algebra libraries. Most of these use sophisticated optimisation of the memory architetcture in order to minimise cache misses, e.g. by block factoring the input matrices before performing an operation. Although my goal was not to compete with these libraries, and despite the present implementation is quite rudimentary in comparison, I was surprised to see that the performances of my custom library are competitive at low dimensions (matrix sizes below 15) and not too bad at higher dimensions - staying within an order of magnitude from the state of the art in terms of average CPU time. Of course this is probably insuficient for performance-critical applications, but not so bad for other applications.

To assess the performances of the present library, the key operations were run using various linear algebra libraries over a range of problem dimensions. For a given operation, library and problem dimension, 100,000 executions of the operation were conducted on random objects and the average time to completion was recorded. The results are shown in the Figure below for low dimensions and for the matrix multiplication operation:

![alt text](https://github.com/martindoff/Cpp-linear-algebra-library/blob/main/img/low_dim.png)

As can be seen from the Figure, the custom library is quite competitive in that range. However, state-of-the-art libraries are taking over at higher dimensions:

![alt text](https://github.com/martindoff/Cpp-linear-algebra-library/blob/main/img/benchmark.png)

## Using compiler optimisation

Compiler optimisation can accelerate computations dramatically. The following flags were added to the Makefile: `-fopenmp` and `-O3`. 

The impact of adding them is shown in the Figure below where a clear improvement is observed by comparison to running the code without compiler optimisation:

![alt text](https://github.com/martindoff/Cpp-linear-algebra-library/blob/main/img/opti.png)

## Choice of data storage

When designing the `Vector` and `Matrix` classes, the `std::vector` container was appealing at first sight as a storage for the data members:
it offered the perpective of delegating memory management of the classes to the container and some other useful operations such as `std::vector::size` or iterators. Problems arose when it came to 
comparing the library performances against state-of-the-art libraries: it seems that the use of the container incurred a substantial overhead. 

I then remembered about the `std::valarray` container, hidden in the depths of the standard library, that was specifically designed for numerical computations. The use of this container is somehow discouraged by the online C++ community - most of the time undeservedly. 
The truth is that most of the C++ developpers criticising the `std::valarray` container do not work on these numerically intensive applications for which it was designed for.

Beyond the many useful numerical operations that it offers (e.g. the possibility to perform arithmetic operations, sum, norm, etc.), it claims to also optimise the way data is accessed. 
Since I had developed my library with the possibility of easily replacing the container used for the data members, I decided it was a good opportunity to see for myself the benefit of `std::valarray` over `std::vector` containers.

And I was not disappointed ! The performance improvement was dramatic, as can be seen on the Figure below:

![alt text](https://github.com/martindoff/Cpp-linear-algebra-library/blob/main/img/valarray.png)

The Figure was obtained by running the exact same algorithms but with different data storage on the same problems. It
appears clearly that `std::valarray` outperforms `std::vector` in terms of CPU time (by about a factor of 3), showing that the latter should 
be favored for computationally intensive tasks. 


## Future work
Future developments could involve improving the library to make it run faster on high dimensional problems or extend its capabilities. For example:
* replacing the container `std::valarray` by plain C-style arrays. We have already seen that the transition from `std::vector` to `std::valarray` allowed to speed up computations by about a factor of 3.  
I would then be curious to see what would be the gain (or pain) resulting in resorting to pure arrays for the data members of the `Matrix` and `Vector` classes. 
* Block decomposition of the input matrices before performing an operation can speed up computations by minimising cache misses. This is an interesting avenue of development for improving the performances of the library.
* Much more linear algebra operations could be added such as a QR decomposition to solve, e.g. least square problems. 
