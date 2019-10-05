#ifndef DEF_MATRIX
#define DEF_MATRIX

#include <vector>
#include <iostream>

class Matrix{
    public:
        Matrix();
        Matrix(int height, int width);
        Matrix(std::vector<std::vector<double>> const &array);
        Matrix(std::vector<double> const &array);

        Matrix operator+(Matrix const &m) const;
        Matrix operator-(Matrix const &m) const;

        // scalar multiplication
        Matrix operator*(double const &value);
        // hadamard product
        Matrix operator*(Matrix const &m) const;
        Matrix dot(Matrix const &m) const;
        Matrix convolution(Matrix const &m) const;

        Matrix transpose() const;
        // to apply a function to every element of the matrix
        Matrix applyFunction(double (*function)(double)) const;

        // pretty print of the matrix
        void print(std::ostream &flux) const;
        int get_rows(){ return height; }
        int get_cols(){ return width; }
        // accessor and setter
        double& operator()(const unsigned& row, const unsigned& col){
            return array[row][col];
        }
        std::vector<std::vector<double>> to_vector(){ return array; }


    private:
        std::vector<std::vector<double> > array;
        int height;
        int width;
};

// overloading << operator to print easily
std::ostream& operator<<(std::ostream &flux, Matrix const &m);

#endif
