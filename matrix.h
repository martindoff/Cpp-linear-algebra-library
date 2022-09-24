#ifndef MATRIX_H
#define MATRIX_H

#include "vector.h"
#include <map>
/**
 *  Matrix
 */ 
template <typename T> class Matrix : public Vector<T>
{
public:
    
    // Import data members and definitions in namespace Vector<T>
    using Vector<T>::data; 
    using Vector<T>::sz;
    using typename Vector<T>::storage_type;
    
    // Ressource management
    Matrix() = default; 
    Matrix(std::initializer_list<std::initializer_list<T>> l);  // initializer list
    Matrix(const size_t nr, const size_t nc, const T& val = T());  // m x n Matrix
    explicit Matrix(const size_t n);  // square Matrix
    Matrix(storage_type, const size_t nr, const size_t nc);  // Matrix from data
    Matrix(const Matrix&) = default;  // copy constructor
    Matrix(Matrix&& m) = default;  // move constructor
    Matrix& operator=(Matrix&& m) = default;  // move operator
    Matrix& operator=(const Matrix&) = default;  // copy assignment operator
    ~Matrix() = default;  // destructor
    
    // Data management
    T& operator()(const size_t, const size_t);  // access elements
    const T& operator()(const size_t, const size_t) const;  // access elements (const)
    Vector<T> get_row(const size_t) const;  // access row
    void set_row(const Vector<T>&, const size_t);  // set row
    void swap(const size_t, const size_t);  // swap rows
    Matrix<T> operator[](std::initializer_list<std::initializer_list<size_t>>) const;  // slice
    size_t nrow()  const { return row; }
    size_t ncol()  const { return col; }
    std::pair<size_t, size_t> shape() const { return std::make_pair(row, col); } 
    
    // Basic usual operations
    Matrix& operator+=(const Matrix&);
    Matrix& operator-=(const Matrix&);
    Matrix& operator*=(const T);
    Matrix& operator/=(const Matrix&);
    Matrix& operator/=(const T);
    const Matrix& operator+() const { return *this; }
    Matrix operator-() const;
    std::ostream& print(std::ostream& s) const;
    
    //Other operations
    
    
    // Linear algebra operations
    Matrix t() const;  // transpose
    Matrix inv() const;  //inverse
    
private:
    size_t row;
    size_t col;
    
};

/**
 *   Non member functions declaration
 */
 
// Create special Matrix
template <typename T> Matrix<T> zeros_like(const Matrix<T>&);
template <typename T> Matrix<T> zeros(const size_t, const size_t);
template <typename T> Matrix<T> eye(const size_t);
template <typename T> Matrix<T> diag(Vector<T>&);
template <typename T> Matrix<T> tril(const size_t);
template <typename T> Matrix<T> rand(const size_t, const size_t, 
                                                        const T& = T(0), const T& = T(1));
template <typename T> Matrix<T> vcat(const Matrix<T>&, const Matrix<T>&);
template <typename T> Matrix<T> hcat(const Matrix<T>&, const Matrix<T>&);

// Basic operations
template <typename T> Matrix<T> operator+(const Matrix<T>&, const Matrix<T>&);
template <typename T> Matrix<T> operator-(const Matrix<T>&, const Matrix<T>&);
template <typename T> Vector<T> operator*(const Matrix<T>&, const Vector<T>&);
template <typename T> Vector<T> operator*(const Vector<T>&, const Matrix<T>&);
template <typename T> Matrix<T> operator*(const Matrix<T>&, const Matrix<T>&);
template <typename T> Matrix<T> operator*(const T, const Matrix<T>&);
template <typename T> Matrix<T> operator*(const Matrix<T>&, const T);
template <typename T> Matrix<T> operator/(const Matrix<T>&, const T);
template <typename T> Matrix<T> operator/(const T, const Matrix<T>&);
template <typename T> Matrix<T> operator^(const Matrix<T>&, const T);
template <typename T> Matrix<T> abs(const Matrix<T>&); 
//template <typename T> Matrix<T> operator-(const Matrix<T>&);  // was tentatively replaced
//template <typename T> std::ostream& operator<<(std::ostream&, const Matrix<T>&);
template <typename T> std::istream& operator>>(std::istream&, Matrix<T>&);

// Logical operations

// Other operations

// Linear algebra operations
template <typename T> T det(const Matrix<T>&);
template <typename T> Matrix<T> LU(const Matrix<T>&); 
template <typename T> std::tuple<Matrix<T>, T, Vector<int>> crout(const Matrix<T>&);
template <typename T> Vector<T> substitute(const Matrix<T>&, const Vector<T>&);
template <typename T> Vector<T> permute(const Vector<T>&, const Vector<int>&); 
template <typename T> Vector<T> operator/(const Matrix<T>&, const Vector<T>&);

 // include implementation to avoid template - related linker errors
//#include "vector.cpp"
#include "matrix.cpp"

#endif