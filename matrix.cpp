#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "matrix.h"
#include "vector.h"
#include <map>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <random>
#include <string>

/**
 * Constructors
 */
 
// Construct m x n Matrix
template <typename T> inline Matrix<T>::Matrix(const size_t nr, const size_t nc, const T& val):
Vector<T>::Vector(nr*nc, val), row{nr}, col{nc} {} 

// Construct square Matrix
template <typename T> inline Matrix<T>::Matrix(const size_t n): 
Vector<T>::Vector(n*n), row{n}, col{n} {}

// Construct Matrix from initialiser list
template <typename T> Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> l): 
Vector<T>::Vector(l.size()*l.begin()->size()),
row{l.size()},
col{l.begin()->size()}
{
    auto i = 0;   
    for (auto iter1 = l.begin(); iter1 != l.end(); ++iter1)
    {
        for (auto iter2 = iter1->begin(); iter2 != iter1->end(); ++iter2)
        {
            this->data[i++] = *iter2;
            // data.push_back(*iter2); 
        }
    }
}

// Construct Matrix from data
template <typename T> Matrix<T>::Matrix(storage_type data, const size_t nr, const size_t nc): 
Vector<T>::Vector(data), row{nr}, col{nc} {}

/**
  * Create special matrices
  */

// Zero Matrix like input Matrix
template <typename T> Matrix<T> zeros_like(const Matrix<T>& m)
{
    return Matrix<T>(m.nrow(), m.ncol(), T(0)); 
}

// Zero Matrix
template <typename T> Matrix<T> zeros(const size_t nr, const size_t nc)
{
    return Matrix<T>(nr, nc, T(0));
}

// Ones Matrix
template <typename T> Matrix<T> ones(const size_t nr, const size_t nc)
{
    return Matrix<T>(nr, nc, T(1));
}

// Identity Matrix
template <typename T> Matrix<T> eye(const size_t n)
{
    Matrix<T> ret(n);
    for (auto i = 0; i != n; ++i)
    {
        ret(i, i) = T(1);
    }
    return ret; 
}

// Diagonal matrix
template <typename T> Matrix<T> diag(Vector<T>& v)
{
    auto n = v.size(); 
    Matrix<T> ret(n);
    for (auto i = 0; i != n; ++i)
    {
        ret(i, i) = v(i);
    }
    return ret;
    
}

// Lower triangular matrix
template <typename T> Matrix<T> tril(const size_t n)
{

    Matrix<T> ret(n);
    for (auto i = 0; i != n; ++i)
    {
        for (auto j = 0; j != n; ++j)
        {
            if (i >= j) { ret(i, j) = T(1); }
        }
    }
    return ret;
    
}

// Random Matrix
template <typename T> Matrix<T> rand(const size_t nr, const size_t nc, 
                                                              const T& mean, const T& var)
{
    Matrix<T> ret(nr, nc);
    
    std::random_device rd{};
    std::mt19937 generator{rd()};
    std::normal_distribution<T> distribution{mean, var};

    for (auto i = 0; i != nr; ++i)
    {
        for (auto j = 0; j != nc; ++j)
        {
            ret(i, j) = distribution(generator);
        }
    }
    return ret; 
}

// Vertical concatenation of 2 Matrix objects

template <typename T> Matrix<T> vcat(const Matrix<T>& a, const Matrix<T>& b)
{
    
    if(a.ncol() != b.ncol())
    {
        throw std::out_of_range("Matrix<T>::vcat. Matrix dimensions mismatch.");
    }
    
    const auto an = a.size();
    const auto anr = a.nrow();
    const auto anc = a.ncol();
    const auto bnr = b.nrow();
    const auto bnc = b.ncol();
    const auto n = an + b.size();
    
    typename Matrix<T>::storage_type dest(n);
    
    auto k = 0;
    
    // Loop through matrix a and add the elements at the end of dest 
    for (auto i = 0; i != anr; ++i)
    {
        for (auto j = 0; j != anc; ++j)
        {
            dest[k++] = a(i, j); 
        }
    }
    
    // Loop through matrix b and add the elements at the end of dest    
    for (auto i = 0; i != bnr; ++i)
    {
        for (auto j = 0; j != bnc; ++j)
        {
            dest[k++] = b(i, j); 
        }
    }
    
    return Matrix<T>(dest, anr+bnr, anc);  
    
}

// Horizontal concatenation of 2 Matrix objects
template <typename T> Matrix<T> hcat(const Matrix<T>& a, const Matrix<T>& b)
{
    if(a.nrow() != b.nrow())
    {
        throw std::out_of_range("Matrix<T>::hcat. Matrix dimensions mismatch.");
    }
       
    const auto an = a.size();
    const auto anr = a.nrow();
    const auto anc = a.ncol();
    const auto bnc = b.ncol();
    const auto nc = anc + bnc;
    const auto n = an + b.size();
     
    
    typename Matrix<T>::storage_type dest(n);
    typename Matrix<T>::storage_type froma = a.get_data();
    typename Matrix<T>::storage_type fromb = b.get_data();
    
    
    for (auto i = 0; i != anr; ++i)
    {
        auto k = 0; 
        for (auto j = 0; j != nc; ++j)
        {
            dest[i*nc + k] = k < anc ? froma[i*anc + k] : fromb[i*bnc + k-anc];
            k++; 
        }
    }
    
    return Matrix<T>(dest, anr, anc + bnc);     
    
}

/**
  * Data management
  */

// Access element
template <typename T> inline T& Matrix<T>::operator()(const size_t i, const size_t j)
{
    const auto stride = col;  // data stored row by row (row major storage) 
                              // e.g. entry  m(1, 1) is followed by m(1, 2) in m.data 
    return data[stride * i + j]; 
}

// Access element (const)
template <typename T> 
inline const T& Matrix<T>::operator()(const size_t i, const size_t j) const
{
    const auto stride = col; 
    return data[stride * i + j];  
}

// Access row
template <typename T> inline Vector<T> Matrix<T>::get_row(const size_t i) const
{
    auto stride = this->ncol(); 
    return Vector<T>(data[std::slice(stride*i, stride, 1)]);
}

// Set row
template <typename T> void Matrix<T>::set_row(const Vector<T>& v, const size_t i)
{
    auto n = this->ncol();
    if (v.size() != n)
    {
        throw std::out_of_range("Matrix<T>::set_row. Matrix dimensions mismatch.");
    }  
    for (auto j = 0; j != n; ++j)
    {
        (*this)(i, j) = v(j);
    }
    
}

// Swap rows
template <typename T> void Matrix<T>::swap(const size_t i, const size_t j)
{
    const auto n = this->ncol();
    //std::swap(data[std::slice(stride*i, stride, 1)], data[std::slice(stride*j, stride, 1)]);
    for (auto k = 0; k != n; ++k)
    {
        std::swap((*this)(i, k), (*this)(j, k));
    }

}

// Slice matrix
template <typename T> Matrix<T> Matrix<T>::operator[]
(std::initializer_list<std::initializer_list<size_t>> l) const
{

    const auto r_begin = *(l.begin()->begin()); 
    const auto r_end = *(l.begin()->end()-1);
    const auto c_begin = *((l.end()-1)->begin()); 
    const auto c_end = *((l.end()-1)->end()-1); 
    
    const auto m = r_end - r_begin;
    const auto n = c_end - c_begin;
    
    if (l.size() != 2 || l.begin()->size() != 2 || (l.end()-1)->size() != 2)
    {
        throw std::domain_error("Matrix<T>::operator[]. Wrong input initializer list");
    }
    if (r_begin >= r_end || c_begin >= c_end || r_end > this->nrow() || c_end > this->ncol())
    {
        throw std::out_of_range("Matrix<T>::operator[]. Slices indices inconsistent");
    }
    
    Matrix<T> ret(m, n);
    
    for (auto i = 0; i != m; ++i)
    {
        for (auto j = 0; j != n; ++j)
        {
            ret(i, j) = (*this)(i + r_begin, j + c_begin); 
        }
    }
    
    return ret;
}


/**
  * Basic algebraic operations
  */
  
// Addition
template <typename T> Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& m)
{
    if(m.nrow() != row || m.ncol() != col)
    {
        throw std::out_of_range("Matrix<T>::operator+=. Matrix dimensions mismatch.");
    }
    this->data += m.get_data();
    return *this;
}

// Subtraction
template <typename T> Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& m)
{
    if(m.nrow() != row || m.ncol() != col)
    {
        throw std::out_of_range("Matrix<T>::operator-=. Matrix dimensions mismatch.");
    }
    this->data -= m.get_data();
    return *this;
}

// Unary minus
template <typename T> inline Matrix<T> Matrix<T>::operator-() const
{
    return Matrix<T>(-this->data, this->nrow(), this->ncol());
}

// Binary addition
template <typename T> Matrix<T> operator+(const Matrix<T>& a, const Matrix<T>& b)
{
    Matrix<T> ret(a);
    return ret += b;
}

// Binary subtraction
template <typename T> Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b)
{
    Matrix<T> ret(a); 
    return ret -= b;
}

// Scalar multiplication
template <typename T> Matrix<T>& Matrix<T>::operator*=(const T val)
{
    this->data *= val;
    return *this;
}

// Matrix division
template <typename T> Matrix<T>& Matrix<T>::operator/=(const Matrix<T>& m)
{
    this->data /= m.get_data();
    return *this;
}

template <typename T> Matrix<T>& Matrix<T>::operator/=(const T val)
{
    this->data /= val;
    return *this;
}

// Scalar-Matrix multiplication
template <typename T> Matrix<T> operator*(const T val, const Matrix<T>& v)
{
    Matrix<T> ret(v); 
    return ret *= val;
   
}

// Matrix-scalar multiplication
template <typename T> Matrix<T> operator*(const Matrix<T>& v, const T val)
{
    Matrix<T> ret(v); 
    return ret *= val;
}

// Matrix-scalar division
template <typename T> Matrix<T> operator/(const Matrix<T>& v, const T val)
{
    Matrix<T> ret(v); 
    return ret /= val;
}

// Scalar-Matrix division
template <typename T> Matrix<T> operator/(const T val, const Matrix<T>& m)
{
    Matrix<T> ret(m.nrow(), m.ncol(), val);
    return ret /= m; 
}

// Matrix-Vector multiplication
template <typename T> Vector<T> operator*(const Matrix<T>& m, const Vector<T>& v)
{
    auto nr = m.nrow();
    auto nc = m.ncol();
    if(nc != v.size())
    {
        throw std::out_of_range("Matrix<T>::operator*. Matrix dimensions mismatch.");
    }
    
    Vector<T> ret(nr, 0.0);
    
    for(auto i = 0; i != nr; ++i)
    {
        for(auto j = 0; j != nc; ++j)
        {
            ret(i) += m(i, j)*v(j); // for i: ret(i) = dot(m.row(i), v); 
        }
    }
    
    return ret;

}

// Vector-Matrix multiplication
template <typename T> Vector<T> operator*(const Vector<T>& v, const Matrix<T>& m)
{
    auto nr = m.nrow();
    auto nc = m.ncol();
    if(v.size() != nr)
    {
        throw std::out_of_range("Matrix<T>::operator*. Matrix dimensions mismatch.");
    }
    
    Vector<T> ret(nc, 0.0);
    
    for(auto i = 0; i != nc; ++i)
    {
        for(auto j = 0; j != nr; ++j)
        {
            ret(i) += m(j, i)*v(j);  // for i: ret(i) = dot(m.col(i), v);
        }
    }
    
    return ret;

}

// Matrix-Matrix multiplication
template <typename T> Matrix<T> operator*(const Matrix<T>& a, const Matrix<T>& b)
{
    auto anr = a.nrow();
    auto anc = a.ncol();
    auto bnr = b.nrow();
    auto bnc = b.ncol();
    if(anc != bnr)
    {
        throw std::out_of_range("Matrix<T>::operator*. Matrix dimensions mismatch.");
    }
    
    Matrix<T> ret(anr, bnc, 0.0);
    
    for(auto i = 0; i != anr; ++i)
    {
        for(auto j = 0; j != bnc; ++j)
        {
            for (auto k = 0; k != bnr; ++k)
            {
                ret(i, j) += a(i, k)*b(k, j); // for i, for j: ret(i, j) 
                                              // = dot(a.row(i), b.col(j)); 
            }
        }
    }
    
    return ret;
}

// Raise to power p
template <typename T> Matrix<T> operator^(const Matrix<T>& m, const T p)
{
    return Matrix<T>(std::pow(m.get_data(), p), m.nrow(), m.ncol()); 
}

// Absolute value
template <typename T> Matrix<T> abs(const Matrix<T>& m)
{
    return Matrix<T>(std::abs(m.get_data()), m.nrow(), m.ncol());   
}

/**
  * Input / Output
  */
  
// Output
template <typename T> std::ostream& Matrix<T>::print(std::ostream& s) const
{
    // Compute an offset to align data
    const auto nr = this->nrow();
    const auto nc = this->ncol();
    const auto prec = 3;
    auto offset = INT_MIN;
    auto m_offset = 0;
    
    for (auto i = 0; i != nr; ++i)
    {
        for (auto j = 0; j != nc; ++j)
        {
            if ((*this)(i, j) < 0)
            { 
                m_offset = log10(std::fabs((*this)(i, j))) + 2;
            }
            else
            {
                m_offset = log10((*this)(i, j)) + 1;
            }
    
            if (m_offset > offset) offset = m_offset;
        }
    }
    
    offset += prec + 1;
    
    // Print aligned
    const auto default_precision = std::cout.precision();  
    std::cout << std::setprecision(prec) << std::fixed << "[";
    for(auto i = 0; i != nr; ++i)
    {
        std::cout << "["; 
        
        for(auto j = 0; j != nc; ++j)
        {
            std::cout << " " << std::setw(offset) << std::right << (*this)(i,j);
        }
        
        if(i != nr - 1) { std::cout << " ]" << std::endl << " "; }
        else { std::cout << " ]]" << std::endl; }
         
    }
    
    std::cout << std::setprecision(default_precision);
    
    return s; 
}

// Input
template <typename T> std::istream& operator>>(std::istream& s, Matrix<T>& m)
{
    const auto nr = m.nrow();
    const auto nc = m.ncol();  
    
    std::cout << "Enter " << nr << " row vectors of size "<< nc << std::endl << std::endl;
     
    for (auto i = 0; i != nr; ++i)
    {
        std::cout << "Enter row vector " << i+1 << std::endl;
        std::vector<T> vec;
        T x; 
        for(auto j = 0; j != nc; ++j)
        {
		    s >> x;
		    m(i, j) = x; 
		
		}
        
    }
    return s;
}

/**
 *  Logical operations
 */

/**
 *  Other operations
 */


/**
 *  Linear algebra operations
 */

// Transpose
template <typename T> Matrix<T> Matrix<T>::t() const
{
 
    Matrix<T> ret(col, row);
    
    for(auto i = 0; i != row; ++i)
    {
        for(auto j = 0; j != col; ++j)
        {
            ret(j, i) = (*this)(i, j); 
        }
    }
    
    return ret;
    
}

// Determinant (via crout algorithm)
template <typename T> T det(const Matrix<T>& m)
{
    if (m.nrow() != m.ncol())  // Determinant exists only for square matrices
    {
        throw std::out_of_range("det. Matrix dimensions mismatch.");
    }
    auto x = crout(m);
    return std::get<1>(x);
    
}

// LU factorization (via crout algorithm)
template <typename T> Matrix<T> LU(const Matrix<T>& m)
{
    if (m.nrow() != m.ncol())  // LU factorization exists only for square matrices
    {
        throw std::out_of_range("LU. Matrix dimensions mismatch.");
    }
    
    auto x = crout(m);
    return std::get<0>(x);
    
}

// Crout algorithm for LU factorization and determinant computation
template <typename T> std::tuple<Matrix<T>, T, Vector<int>> crout(const Matrix<T>& m)
{
    Matrix<T> ret(m);
    const auto n = m.nrow(); 
    T det{1};
    std::map<int, int> index_table;  // associative container to hold a record of row swaps
    Vector<int> p = range<int>(n);  // vector of permutations 
    T element_diag;
    static constexpr T tol_crout{1e-12};  // tol. of Crout algo for division by diagonal elements
    
    for (auto i = 0; i != n; ++i)
    {
        
        // Row pivot (potentially swap current row with row of maximum pivot element)
        T element_max{0};
        auto index_max = i;
        T det_sign{1};  
        
        for (auto l = i; l != n; ++l)
        {
            auto element_current = std::fabs(ret(l, i));
            if (element_max < element_current) 
            {
                element_max = element_current;
                index_max = l; 
            }
        }
        if (index_max > i)  // if a row was found to replace the current one
        { 
            
            ret.swap(i, index_max);  // swap rows
            std::swap(p(i), p(index_max));  // associate a new position to swapped row
            det_sign = -det_sign;  // change sign of determinant
        }
        
        // Populate the Crout matrix
        for (auto j = 0; j != n; ++j)
        {
    
            for (auto k = 0; k < std::min(i, j) ; ++k)
            {
                ret(i, j) -= ret(i, k)*ret(k, j);
            }
            
            // Division of the upper triangular part by the corresponding diagonal element
            element_diag = ret(i, i);
            if (j > i)
            {

                if (std::fabs(element_diag) < tol_crout)
                {
                    throw std::overflow_error("crout. Division by" 
                                                        " diagonal element < tol_crout.");
                }
                
                ret(i, j) /= element_diag;  
            }
        }
        
        // The determinant of a triangular matrix is the product of the diagonal elements
        det *= det_sign > 0 ? element_diag : -element_diag;  // apply swap sign change
         
    }
    return std::make_tuple(ret, det, p); 
}

// Perform successive forward and backward substitutions to solve Ly = b and Ux = y
template <typename T> Vector<T> substitute(const Matrix<T>& lu, const Vector<T>& b)
{
    const auto n = lu.nrow(); 
    if(n != b.size() || n != lu.ncol())
    {
        throw std::out_of_range("substitute. Matrix dimensions mismatch.");    
    }
    
    Vector<T> x{b};  // x will successively carry b and y in the same memory space
    static constexpr T tol_crout{1e-12};  // tol. of Crout algo for division by diagonal elements  
    
    // Solve Ly = b by forward substitution
    for (auto i = 0; i != n; ++i)
    {
        for (auto j = 0; j != i; ++j)
        {
            x(i) -= lu(i, j)*x(j);  
        }
        auto element_diag = lu(i, i);
        if (std::fabs(element_diag) < tol_crout)
        {
            throw std::overflow_error("substitute. Division by"
                                                        " diagonal element < tol_crout.");
        }
        x(i) /= element_diag; 
    }
    
    // Solve Ux = y by backward substitution
    for (auto i = n-1; i != -1; --i)
    {
        for (auto j = i+1; j != n; ++j)
        {
            x(i) -= lu(i, j)*x(j);  
        }
    }
    
    return x; 

}

// Swap elements of a vector according to a permutation table
template <typename T> Vector<T> permute(const Vector<T>& v, const Vector<int>& p)
{
   const auto n = v.size();
   Vector<T> ret{v};
   
   for (auto i = 0; i != n; ++i)
   {
       ret(i) = v(p(i));
   }
   
   return ret; 
}

// Solve linear system A x = b (via x = A/b)
template <typename T> Vector<T> operator/(const Matrix<T>& a, const Vector<T>& b)
{
    if(a.nrow() != b.size())
    {
        throw std::out_of_range("operator/. Matrix dimensions mismatch.");    
    }
    
    if (a.nrow() == a.ncol())  // Square matrix: LU decomposition
    {
        // LU factorization of A. L & U stored in a single matrix A_LU. 
        auto [lu, det, p] = crout(a);
        
        // Apply permutations to b following crout partial pivoting
        Vector<T> b_swap = permute(b, p);
        
        // Forward and backward substitution
        return substitute(lu, b_swap);
        
    }
    else if (a.nrow() >= a.ncol())  // Overdetermined: QR decomposition
    {
        // QR_factorize()
        // substitute
        throw std::out_of_range("Matrix<T>::operator/. Attempting to solve A x = b "
        "but matrix A is tall. Problem overdetermined.");

    }
    else  // Underdetermined: QR decomposition
    {
        throw std::out_of_range("Matrix<T>::operator/. Attempting to solve A x = b "
        "but matrix A is flat. Problem underdetermined.");
    }
}

// Inverse
template <typename T> Matrix<T> Matrix<T>::inv() const
{
    const auto n = this->nrow();
    if (n != this->ncol())
    {
        throw std::out_of_range("inv. Matrix dimensions mismatch."); 
    }
    
    Matrix<T> I = eye<T>(n);
    static constexpr T tol_crout{1e-12};  // tol. of Crout algo for division by diagonal elements
    
    // LU decomposition
    auto [lu, det, p] = crout(*this);
    
    // Check determinant
    if (std::fabs(det) < tol_crout)
    {
        throw std::overflow_error("inv. Matrix singular to working precision."); 
    }
    
    // Solve A x = e where e are the row vectors of the identity matrix
    for (auto i = 0; i != n; ++i)
    {
        Vector<T> b_swap = permute(I.get_row(i), p);
        I.set_row(substitute(lu, b_swap), i);  // get x as a row and store it in row # i 
    }
    
    return I.t(); 
}
 
#endif 