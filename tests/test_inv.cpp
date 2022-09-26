#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include "matrix.h"

template <typename T> Matrix<T> Eigen2Matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>);
template <typename T> Vector<T> Eigen2Vector(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>);

template <typename T> Matrix<T> Eigen2Matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m)
{
    auto n_row = m.rows(); 
    auto n_col = m.cols(); 
    
    Matrix<T> ret(n_row, n_col);
    
    for (auto i = 0; i != n_row; ++i)
    {
        for (auto j = 0; j != n_col; ++j)
        {
            ret(i, j) = m(i, j); 
        }
    }
    
    return ret; 
}

template <typename T> Vector<T> Eigen2Vector(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> v)
{
    auto n = v.rows(); 
    
    Vector<T> ret(n);
    
    for (auto i = 0; i != n; ++i)
    {
        ret(i) = v(i);
    }
    
    return ret; 
}
int main()
{
constexpr int N_experiment = 10000; 
constexpr int N = 20;  // dimension of matrix 
auto times_custom = std::chrono::microseconds::zero().count();
auto times_eig = times_custom; 
int count = 0; 
double tol = 1e-6;  // tolerance for comparison error
double inf = std::numeric_limits<double>::infinity();

for (auto i = 0; i != N_experiment; ++i)
{ 
    // Generate a random Eigen matrix 
    Eigen::Matrix<double, N, N> m = Eigen::Matrix<double, N, N>::Random();

    // Convert Eigen matrix to custom Matrix
    Matrix<double> A = Eigen2Matrix<double>(m);
     
    // Compute with custom method
    auto t0 = std::chrono::high_resolution_clock::now(); 
    Matrix<double> A_inv = A.inv();
    auto t1 = std::chrono::high_resolution_clock::now();
    times_custom += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
   
    // Compute using Eigen method
    t0 = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<double, N, N> m_inv = m.inverse();
    t1 = std::chrono::high_resolution_clock::now();
    times_eig += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();

    // Compare 
    Matrix<double> m_inv_ = Eigen2Matrix<double>(m_inv);  // conversion from Eigen
    double err = norm(A_inv - m_inv_, inf);
    
    if (err < tol) {count++;}
    else {std::cout << "Comparison error above specified tolerance:" << err << std::endl;}
}

std::cout << "Loop terminated with " << count << " successes out of " 
          << N_experiment << " experiments\n"
          << "Average time per call for custom method: " 
          << times_custom/double(N_experiment) << " microseconds" << std::endl
          << "Average time per call for eigen method: " 
          << times_eig/double(N_experiment) << " microseconds" << std::endl;  
return 0; 
}