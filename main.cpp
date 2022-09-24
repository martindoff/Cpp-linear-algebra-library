#include <iostream>
#include <limits>
#include "vector.h"
#include "matrix.h"
 
//using Eigen::MatrixXd;
 
 

int main()
{
/*  Vector  */
Vector<double> a = {1, 2, 3};
Vector<double> b = zeros<double>(6);
Vector<double> c = zeros_like(b);
Vector<double> d = ones<double>(2);
Vector<double> e = ones_like(d);
Vector<double> f = rand(4, 10.0, 2.0);
Vector<double> g(a);
Vector<double> h = {8, 6, 4, 2};

std::cout << "************************************************************************\n";
std::cout << "***************************** Vector test ******************************\n";
std::cout << "************************************************************************\n";
std::cout << std::endl;

std::cout << "Print some vectors\n";
std::cout << "------------------\n";
std::cout << "a = " << a; 
std::cout << "b = "<<b;
std::cout << "c = "<<c;
std::cout << "d = "<<d;
std::cout << "e = "<<e;
std::cout << "f = "<<f; 
std::cout << "g = "<<g;
std::cout << "h = "<<h;
std::cout << std::endl;

std::cout << "Arithmetic operations on vectors (a = [1 2 3])\n";
std::cout << "----------------------------------------------\n";
std::cout << "a + ones_like(a) = " << a + ones_like(a);
std::cout << "a - ones_like(a) = " << a - ones_like(a);
std::cout << "-a = " << -a; 
std::cout << "+a = " << +a;
std::cout << "a^2.0 = " << (a^2.0);
std::cout << "a*2.0 = " << a*2.0;
std::cout << "1.0*a = " << 1.0*a;
std::cout << "a/3.0 = " << a/3.0;
std::cout << "1/a = " << 1.0/a;
std::cout << "a * {2, 0, 1} = " << a * Vector<double>({2, 0, 1});
std::cout << "a / {2, 0, 1} = " << a / Vector<double>({2, 0, 1});
std::cout << std::endl;

std::cout << "Access elements of vectors (a = [1 2 3])\n";
std::cout << "----------------------------------------\n";
std::cout << "a[0] = " << a(0) << std::endl;
std::cout << "a[1] = " << a(1) << std::endl;
std::cout << "a.size() = " << a.size() << std::endl;
std::cout << "a[{0, 2}] = "<< a[{0, 2}];
std::cout << "a.swap(0, 2): "; a.swap(0, 2);
std::cout << "a = " << a; 
std::cout << "a-=4.0: a= " << (a-=4.0) << "a*=-1.0: a = " << (a*=-1.0);
std::cout << "cat({1, 2}, {3, 4, 5}) = " << cat(Vector<double>({1, 2}), Vector<double>({3, 4, 5})); 
std::cout << std::endl;

std::cout << "Measurement operations on vectors (a = [1 2 3])\n";
std::cout << "-----------------------------------------------\n";
std::cout << "abs(-a) = " << abs(-a);
std::cout << "min(a) = " << min(a) << std::endl;
std::cout << "max(a) = " << max(a) << std::endl;
std::cout << "sum(a) = " << sum(a) << std::endl;
std::cout << "norm(a) = " << norm(a)<< std::endl;
std::cout << "norm(a, 1.0) = " << norm(a, 1.0)<< std::endl;
std::cout << "norm(a, inf) = "<< norm(a,std::numeric_limits<double>::infinity())<<std::endl;
std::cout << std::endl; 

std::cout << "Other operations on vectors (a = [1 2 3])\n";
std::cout << "------------------------------------------\n";
std::cout << "diff({4, 6, 10, -9, 5}) = " << diff(Vector<double>({4, 6, 10, -9, 5}));
std::cout << "cumsum(a) = " << cumsum(a);
std::cout << "cumsum(a, 'reverse') = " << cumsum(a, "reverse");
std::cout << "dot({1, 1, 0}, a) = " << dot(Vector<double>({1, 1, 0}), a) << std::endl;
std::cout << std::endl;

/*  Matrix  */
std::cout << "************************************************************************\n";
std::cout << "***************************** Matrix test ******************************\n";
std::cout << "************************************************************************\n";
std::cout << std::endl;

Matrix<double> A(2, 3, 4.0);
Matrix<double> B(A);
Matrix<double> C(2);
Matrix<double> D = {{1, 2, 3}, {0, 0, 1}, {1, 1, 1}};
Matrix<double> E(2, 2, 1.0);
Matrix<double> F = ones<double>(2, 3);
Matrix<double> G = eye<double>(3);
Matrix<double> H = {{1, 2, 3, 4}, {2, 3, 4, 6}, {3, 4, 2, 5}, {4, 6, 5, 7}}; 
std::cout << std::endl;

std::cout << "Print some matrices\n";
std::cout << "-------------------\n";
std::cout << "A(2, 3, 4.0) = \n" << A;
std::cout << "B(A) = \n" << B;
std::cout << "C(2) = \n" << C;
std::cout << "D = {{1, 2, 3}, {0, 0, 1}, {1, 1, 1}} = \n" << D;
std::cout << "E(2, 2, 1.0) = \n" << E;
std::cout << "F = ones<double>(2, 3) = \n" << F;
std::cout << "G = eye<double>(3) = \n" << G;
std::cout << "H = {{1, 2, 3, 4}, {2, 3, 4, 6}, {3, 4, 2, 5}, {4, 6, 5, 7}} = \n" << G;
std::cout << "diag({1, 2, 3}) =\n" << diag(a);
std::cout << "tril<double>(4) =\n" << tril<double>(4);
std::cout << std::endl;

std::cout << "Basic algebraic operations (D = {{1, 2, 3}, {0, 0, 1}, {1, 1, 1}})\n";
std::cout << "------------------------------------------------------------------\n";
std::cout << "-D =\n" << -D;
std::cout << "+D =\n" << +D;
std::cout << "D + D =\n" << D + D;
std::cout << "3.0*D =\n" << 3.0*D;
std::cout << "D/2.0 =\n" << D/2.0;
std::cout << "1.0/D =\n" << 1.0/D;
std::cout << "D^2.0 =\n" << (D^2.0);
std::cout << std::endl;

std::cout << "Access elements of a matrix \n";
std::cout << "----------------------------\n";
std::cout << "B.size(), B.nrow(), B.ncol(): ";
std::cout << B.size() << " " << B.nrow() << " " << B.ncol() << "\n";
std::cout << "B = E, B.size(), B.nrow(), B.ncol(): ";
B = E;
std::cout << B.size() << " " << B.nrow() << " " << B.ncol() << "\n";

std::cout << "Print D element by element (D = {{1, 2, 3}, {0, 0, 1}, {1, 1, 1}})\n";
std::cout << D(0, 0) << " " << D(0, 1) << " " <<  D(0, 2) << " " << std::endl
          << D(1, 0) << " " << D(1, 1) << " " <<  D(1, 2) << " " << std::endl
          << D(2, 0) << " " <<  D(2, 1) << " " << D(2, 2) << " " << std::endl;
std::cout << "A.shape(): ";
auto [row, col] = A.shape(); 
std::cout << "(" << row << ", "<< col << ")" << std::endl;
std::cout << "G =\n" << G;
std::cout << "G(0, 0) = 0; "; 
G(0, 0) = 0;
std::cout << "G =\n" << G;
std::cout << "r = G.get_row(1); ";
Vector<double> r = G.get_row(1);  
std::cout << "r(0), r(1), r(2): " << r(0) << " " << r(1) << " " << r(2) << " " << std::endl;
std::cout << "G.set_row({1, 1, 3}, 0); ";
G.set_row({1, 1, 3}, 0);
std::cout << "G =\n" << G;
std::cout << "G.swap(0, 2); ";
G.swap(0, 2);
std::cout << "G =\n" << G;
std::cout << "D[{{0, 2}, {0, 3}}] =\n" << D[{{0, 2}, {0, 3}}];
std::cout << "vcat(A, D) = \n" << vcat(A, D);
std::cout << "hcat({{1, 2}, {6, 7}}, {{3, 4, 5}, {8, 9, 10}}) = \n" 
          << hcat(Matrix<double>({{1, 2}, {6, 7}}), Matrix<double>({{3, 4, 5}, {8, 9, 10}}));
std::cout << std::endl;


std::cout << "Matrix measurement operations \n";
std::cout << "------------------------------\n";
std::cout << "abs(-D) =\n" << abs(-D);
std::cout << "min(D) = " << min(D) << std::endl;
std::cout << "max(D) = " << max(D) << std::endl;
std::cout << "sum(D) = " << sum(D) << std::endl;
std::cout << "norm(D) = " << norm(D)<< std::endl;
std::cout << "norm(D, 1.0) = " << norm(D, 1.0)<< std::endl;
std::cout << "norm(D, inf) = "<< norm(D,std::numeric_limits<double>::infinity())<<std::endl;
std::cout << std::endl;

std::cout << "Linear algebra operations \n";
std::cout << "--------------------------\n"; 
std::cout << "{{1, 2}, {0, 1}} * {1, 2} =\n" 
          << Matrix<double>({{1, 2}, {0, 1}}) * Vector<double>({1, 2});
std::cout << "{1, 2} * {{1, 2}, {0, 1}} =\n" 
          << Vector<double>({1, 2}) * Matrix<double>({{1, 2}, {0, 1}}); 
std::cout << "{{1, 2, 3}, {3 6 9}} * D =\n" << Matrix<double>({{1, 2, 3}, {3, 6, 9}}) * D;
std::cout << "H =\n" << H;
std::cout << "H/h = " << H / h; 
std::cout << "det(H) = " << det(H) << std::endl;
std::cout << "H.inv() =\n" << H.inv();
std::cout << std::endl;

return 0; 
}