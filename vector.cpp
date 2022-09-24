#ifndef VECTOR_CPP
#define VECTOR_CPP

#include "vector.h"
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <random>
#include <string>
#include <climits>

/**
 * Vector constructors
 */
 
// Construct Vector of size n
template <typename T> inline Vector<T>::Vector(const size_t n, const T& val): sz{n}
{
    data.resize(n, val);
}

// Construct Vector from initialiser list
template <typename T> Vector<T>::Vector(std::initializer_list<T> l): 
sz{l.size()}
{
    data.resize(sz);
    
    for (auto iter = std::make_pair(l.begin(), std::begin(this->data)); 
                                      iter.first != l.end(); ++iter.first, ++iter.second)
    {
        *(iter.second) = *(iter.first);
    }
}

// Construct Vector from data
template <typename T> Vector<T>::Vector(storage_type store): 
sz{store.size()}, data(store) {}

/**
  * Create special Vectors
  */

// Zero Vector like input Vector
template <typename T> Vector<T> zeros_like(const Vector<T>& v)
{
    return Vector<T>(v.size(), T(0)); 
}

// Vector of ones like input Vector
template <typename T> Vector<T> ones_like(const Vector<T>& v)
{
    return Vector<T>(v.size(), T(1)); 
}

// Zero Vector
template <typename T> Vector<T> zeros(const size_t n)
{
    return Vector<T>(n, T(0));
}

// Ones Vector
template <typename T> Vector<T> ones(const size_t n)
{
    return Vector<T>(n, T(1));
}

// Create a range Vector [0, 1, 2 ... n-1]
template <typename T> Vector<T> range(const size_t n)
{
    Vector<T> v(n);
    for (auto i = 0; i!= n; ++i)
    {
        v(i) = i; 
    }

    return v; 
}

// Random Vector
template <typename T> Vector<T> rand(const size_t n, const T& mean, const T& var)
{
    Vector<T> ret(n);
    
    std::random_device rd{};
    std::mt19937 generator{rd()};
    std::normal_distribution<T> distribution{mean, var};

    for (auto i = 0; i != n; ++i)
    {
        ret(i) = distribution(generator);
    }
    return ret; 
}

template <typename T> Vector<T> cat(const Vector<T>& a, const Vector<T>& b)
{
    
    const auto an = a.size();
    const auto n = an + b.size();
     
    typename Vector<T>::storage_type dest(n);
    for (auto i = 0; i != n; ++i)
    {
        dest[i] = i < an ? a(i) : b(i-an); 
    }
    
    return Vector<T>(dest);
} 
/**
  * Data management
  */

// Access element
template <typename T> inline T& Vector<T>::operator()(const size_t i)
{
    return data[i];  // check for bad access
}

// Access element (const)
template <typename T> 
inline const T& Vector<T>::operator()(const size_t i) const
{
    return data[i];  
}

// Slice
template <typename T> Vector<T> Vector<T>::operator[](std::initializer_list<size_t> l) const
{
    if (l.size() != 2)
    {
        throw std::domain_error("Vector<T>::operator[]. Wrong input initializer list");
    }
    
    auto i = *(l.begin());
    auto j = *(l.end()-1);
    if (i >= j || j > this->size())
    {
        throw std::out_of_range("Vector<T>::operator[]. Slices indices inconsistent");
    }
    
    return Vector<T>(data[std::slice(i, j-i, 1)]);
}

// Swap elements
template <typename T> inline void Vector<T>::swap(const size_t i, const size_t j)
{
    std::swap((*this)(i), (*this)(j));
}
/**
  * Basic algebraic operations
  */
  
// Addition
template <typename T> Vector<T>& Vector<T>::operator+=(const Vector<T>& v)
{
    if(v.size() != sz)
    {
        throw std::out_of_range("Vector<T>::operator+=. Vector dimensions mismatch.");
    }
    this->data += v.get_data();
    return *this;
}

// Subtraction
template <typename T> Vector<T>& Vector<T>::operator-=(const Vector<T>& v)
{
    if(v.size() != sz)
    {
        throw std::out_of_range("Vector<T>::operator+=. Vector dimensions mismatch.");
    }

    this->data -= v.get_data();
    return *this;
}


// Unary minus
template <typename T> inline Vector<T> Vector<T>::operator-() const
{  
    return Vector<T>(-this->data);
}

// Multiplication
template <typename T> Vector<T>& Vector<T>::operator*=(const Vector<T>& v)
{
    if(v.size() != sz)
    {
        throw std::out_of_range("Vector<T>::operator*=. Vector dimensions mismatch.");
    }

    this->data *= v.get_data();
    return *this;

}

// Addition with scalar
template <typename T> Vector<T>& Vector<T>::operator+=(const T val)
{

    this->data += val;
    return *this;
}

// Subtraction with scalar
template <typename T> Vector<T>& Vector<T>::operator-=(const T val)
{

    this->data -= val;
    return *this;
}

// Multiplication by scalar
template <typename T> Vector<T>& Vector<T>::operator*=(const T val)
{
    
    this->data *= val;
    return *this;

}

// Division
template <typename T> Vector<T>& Vector<T>::operator/=(const Vector<T>& v)
{
    if(v.size() != sz)
    {
        throw std::out_of_range("Vector<T>::operator/=. Vector dimensions mismatch.");
    }

    this->data /= v.get_data();
    return *this;

}

// Division by scalar
template <typename T> Vector<T>& Vector<T>::operator/=(const T val)
{

    this->data /= val;
    return *this;

}

// Binary addition 
template <typename T> Vector<T> operator+(const Vector<T>& a, const Vector<T>& b)
{

    Vector<T> ret(a); 
    return ret += b;
}

// Binary subtraction
template <typename T> Vector<T> operator-(const Vector<T>& a, const Vector<T>& b)
{
    Vector<T> ret(a); 
    return ret -= b;
}

// Binary multiplication
template <typename T> Vector<T> operator*(const Vector<T>& a, const Vector<T>& b)
{

    Vector<T> ret(a); 
    return ret *= b;
}

// Scalar-Vector addition 
template <typename T> Vector<T> operator+(const T val, const Vector<T>& a)
{

    Vector<T> ret(a); 
    return ret += val;
}

// Vector-scalar addition
template <typename T> Vector<T> operator+(const Vector<T>& a, const T val)
{
    Vector<T> ret(a); 
    return ret += val;
}

// Scalar-Vector subtraction
template <typename T> Vector<T> operator-(const T val, const Vector<T>& a)
{
    Vector<T> ret(a); 
    return ret -= val;
}

// Vector-scalar subtraction
template <typename T> Vector<T> operator-(const Vector<T>& a, const T val)
{
    Vector<T> ret(a); 
    return ret -= val;
}

// Scalar-Vector multiplication
template <typename T> Vector<T> operator*(const T val, const Vector<T>& v)
{
    Vector<T> ret(v); 
    return ret *= val;
   
}

// Vector-scalar multiplication
template <typename T> Vector<T> operator*(const Vector<T>& v, const T val)
{
    Vector<T> ret(v); 
    return ret *= val;
}

// Binary division
template <typename T> Vector<T> operator/(const Vector<T>& a, const Vector<T>& b)
{

    Vector<T> ret(a); 
    return ret /= b;
}

// Vector-scalar division
template <typename T> Vector<T> operator/(const Vector<T>& v, const T val)
{
    Vector<T> ret(v); 
    return ret /= val;
}

// Scalar-Vector division
template <typename T> Vector<T> operator/(const T val, const Vector<T>& v)
{
    Vector<T> ret(v.size(), val);
    return ret /= v; 
}

// Raise to power p
template <typename T> Vector<T> operator^(const Vector<T>& v, const T p)
{

    return Vector<T>(std::pow(v.get_data(),p));
}

// Absolute value
template <typename T> Vector<T> abs(const Vector<T>& v)
{
    return Vector<T>(std::abs(v.get_data()));  
}

/**
  * Input / Output
  */
  
// Output 
template <typename T> std::ostream& operator<<(std::ostream& s, const Vector<T>& v)
{
    return v.print(s); 
}
template <typename T> std::ostream& Vector<T>::print(std::ostream& s) const
{
    // Compute an offset to align data
    const auto n = this->size(); 
    const auto prec = 3;
    auto offset = INT_MIN;
    auto m_offset = 0;
    for (auto i = 0; i != n; ++i)
    {
        if ((*this)(i) < 0)
        { 
            m_offset = log10(std::fabs((*this)(i))) + 2;
        }
        else
        {
            m_offset = log10((*this)(i)) + 1;
        }
    
        if (m_offset > offset) offset = m_offset;
    }
    
    offset += prec + 1;
    
    // Print aligned
    const auto default_precision = std::cout.precision();  
    std::cout << std::setprecision(prec) << std::fixed << "[ ";
    for(auto i = 0; i != n; ++i)
    {       
        std::cout << std::setw(offset) << std::right << (*this)(i) << " ";
         
    }
    std::cout << "]" << std::endl;
    std::cout << std::setprecision(default_precision);
    
    return s; 
}

// Input
template <typename T> std::istream& operator>>(std::istream& s, Vector<T>& v)
{
    const auto n = v.size();
    T x;
    
    std::cout << "Enter one " << " vector of size "<< n << std::endl << std::endl;
    
    for (auto i = 0; i != n; ++i)
    {
        s >> x;
		v(i) = x; 
        
    }
    return s;
}

/**
 *  Other operations
 */
 
// Sum
template <typename T> T sum(const Vector<T>& v)
{
    return v.get_data().sum(); 
}

// Minimum
template <typename T> T min(const Vector<T>& v)
{
    return v.get_data().min();  
}

// Maximum
template <typename T> T max(const Vector<T>& v)
{
    return v.get_data().max();  
}


// Norm
template <typename T> T norm(const Vector<T>& v, const T p)
{
    if (p <= 0)
    {
        throw std::domain_error("Invalid norm subscript p: must be strictly positive");
    }
    else if (p == 1)   // compute 1-norm 
    {
        auto result = T(0);
        const auto n = v.size();
        for (auto i = 0; i != n; ++i)
        {
            result += std::fabs(v(i)); 
        }
        
        return result; 
    }
    else if (std::isinf(p))  // compute inf-norm
    {
  
        auto result = v(0);
        const auto n = v.size(); 
        for (auto i = 0; i != n; ++i)
        { 
            result = std::max(result, std::fabs(v(i))); 
        }
        
        return result;
    }
    else  // compute p-norm 
    {
     
        auto result = T(0);
        const auto n = v.size();
        for (auto i = 0; i != n; ++i)
        {
            result += std::pow(v(i), p); 
        }
        
        return std::pow(result, 1/p);
    }
}

// Dot product of two vectors
template <typename T> T dot(const Vector<T>& a, const Vector<T>& b)
{
    if(a.size() != b.size())
    {
        throw std::out_of_range("dot. Vector dimensions mismatch.");
    }
    
    return sum(a*b);

}

// Cumulative sum

template <typename T> Vector<T> cumsum(const Vector<T>& v, const std::string& s)
{
    Vector<T> ret(v);
    const auto n = v.size();
    T count{0};
    
    if (s == "forward")
    {
        for (auto i = 0; i != n; ++i)
        {
            ret(i) = count += v(i); 
        }
        return ret;
    }
    else if (s == "reverse")
    {
        for (auto i = n-1; i != -1; --i)
        {
            ret(i) = count += v(i); 
        }
        return ret;
    }
    else
    {
        throw std::domain_error("Invalid direction for cumulative sum.");
    }
}

// Differentiation
template <typename T> Vector<T> diff(const Vector<T>& v)
{
//Initially, v = 4 6 9 13 18 19 19 15 10 
//Modified v = 2 3 4 5 1 0 -4 -5 
    const auto n = v.size();
    return v[{1, n}] - v[{0, n-1}];
}

#endif