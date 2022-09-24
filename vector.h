#ifndef VECTOR_H
#define VECTOR_H

#include <valarray>
#include <initializer_list>
#include <cstddef>
#include <iostream>
#include <string>
#include <algorithm>

/**
 *  Vector class
 */
template <typename T> class Vector
{
public:
    // Some definitions
    using value_type = T;
    using storage_type = typename std::valarray<T>; 
    
    // Ressource management
    Vector() = default; 
    explicit Vector(const size_t n, const T& val = T());  // n vector
    Vector(std::initializer_list<T> l);  // initializer list
    Vector(storage_type);  // Vector from data
    Vector(const Vector&) = default;  // copy constructor
    Vector(Vector&& m) = default;  // move constructor
    Vector& operator=(Vector&& m) = default;  // move operator
    Vector& operator=(const Vector&) = default;  // copy assignment operator
    ~Vector() = default;  // destructor
    

    
    // Data management
    T& operator()(const size_t);  // access elements
    const T& operator()(const size_t) const;  // access elements (const)
    Vector<T> operator[](std::initializer_list<size_t>) const;  // slice vector
    size_t size() const { return data.size(); }
    storage_type get_data() const { return data; }  // replace every instance of this by friends
    void swap(const size_t, const size_t);
    
    // Basic usual operations
    Vector& operator+=(const Vector&);
    Vector& operator+=(const T);
    Vector& operator-=(const Vector&);
    Vector& operator-=(const T);
    Vector& operator*=(const Vector&);
    Vector& operator*=(const T);
    Vector& operator/=(const Vector&);
    Vector& operator/=(const T);
    const Vector& operator+() const { return *this; }
    Vector operator-() const;
    virtual std::ostream& print(std::ostream& s) const;
    
protected:
    size_t sz; 
    storage_type data;
}; 

/**
 *   Non member functions declaration
 */

// Create special Vector
template <typename T> Vector<T> zeros_like(const Vector<T>&);
template <typename T> Vector<T> ones_like(const Vector<T>&);
template <typename T> Vector<T> zeros(const size_t);
template <typename T> Vector<T> ones(const size_t);
template <typename T> Vector<T> rand(const size_t, const T& = T(0), const T& = T(1));
template <typename T> Vector<T> range(const size_t);
template <typename T> Vector<T> cat(const Vector<T>&, const Vector<T>&);

// Basic operations
template <typename T> Vector<T> operator+(const Vector<T>&, const Vector<T>&);
template <typename T> Vector<T> operator+(const T, const Vector<T>&);
template <typename T> Vector<T> operator+(const Vector<T>&, const T);
template <typename T> Vector<T> operator-(const Vector<T>&, const Vector<T>&);
template <typename T> Vector<T> operator-(const T, const Vector<T>&);
template <typename T> Vector<T> operator-(const Vector<T>&, const T);
template <typename T> Vector<T> operator*(const Vector<T>&, const Vector<T>&);
template <typename T> Vector<T> operator*(const T, const Vector<T>&);
template <typename T> Vector<T> operator*(const Vector<T>&, const T);
template <typename T> Vector<T> operator/(const Vector<T>&, const Vector<T>&);
template <typename T> Vector<T> operator/(const Vector<T>&, const T);
template <typename T> Vector<T> operator/(const T, const Vector<T>&);
template <typename T> Vector<T> operator^(const Vector<T>&, const T);
template <typename T> Vector<T> abs(const Vector<T>&); 
template <typename T> std::ostream& operator<<(std::ostream&, const Vector<T>&);
template <typename T> std::istream& operator>>(std::istream&, Vector<T>&);

// Other operations
template <typename T> T sum(const Vector<T>&);
template <typename T> T min(const Vector<T>&);
template <typename T> T max(const Vector<T>&);
template <typename T> T norm(const Vector<T>&, const T = T(2));
template <typename T> T dot(const Vector<T>&, const Vector<T>&); 
template <typename T> Vector<T> cumsum(const Vector<T>&, const std::string& = "forward");
template <typename T> Vector<T> diff(const Vector<T>&);
 // include implementation to avoid template - related linker errors
#include "vector.cpp"

#endif