#include "../include/matrix.hpp"
#include <assert.h>
#include <sstream>

Matrix::Matrix(){}

Matrix::Matrix(int height, int width){
    this->height = height;
    this->width = width;
    this->array = std::vector<std::vector<double> >(height, std::vector<double>(width));
}

Matrix::Matrix(std::vector<std::vector<double> > const &array){
    assert(array.size()!=0);
    this->height = array.size();
    this->width = array[0].size();
    this->array = array;
}

Matrix::Matrix(std::vector<double> const &array){
    assert(array.size()!=0);
    this->height = 1;
    this->width = array.size();
    this->array = {array};
}

Matrix Matrix::operator*(double const &value){
    Matrix result(height, width);
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            result.array[i][j] = array[i][j] * value;
        }
    }
    return result;
}

Matrix Matrix::operator+(Matrix const &m) const{
    assert(height==m.height && width==m.width);
    Matrix result(height, width);
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            result.array[i][j] = array[i][j] + m.array[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(Matrix const &m) const{
    assert(height==m.height && width==m.width);
    Matrix result(height, width);
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            result.array[i][j] = array[i][j] - m.array[i][j];
        }
    }
    return result;
}

// perform hadamard product
Matrix Matrix::operator*(Matrix const &m) const{
    assert(height==m.height && width==m.width);
    Matrix result(height, width);
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            result.array[i][j] = array[i][j] * m.array[i][j];
        }
    }
    return result;
}

Matrix Matrix::dot(Matrix const &m) const{
    assert(width==m.height);
    int mwidth = m.width;
    double w=0;

    Matrix result(height, mwidth);
    for (int i = 0; i < height; i++){
        for (int j = 0; j < mwidth; j++){
            for (int h = 0; h<width; h++){
                w += array[i][h]*m.array[h][j];
            }
            result.array[i][j] = w;
            w=0;
        }
    }
    return result;
}

Matrix Matrix::convolution(Matrix const &mat) const{
    Matrix result(width-1, height-1);
    for(int i = 0; i < result.width; i++){
        for(int j = 0; j < result.height; j++){
            for(int m = 0; m < mat.width; m++){
                for(int n = 0; n < mat.height; n++){
                    result(i, j) += array[i+m][i+n] * mat.array[m][n];
                }
            }
        }
    }
    return result;
}

Matrix Matrix::transpose() const{
    Matrix result(width, height);
    for (int i=0; i<width; i++){
        for (int j=0; j<height; j++){
            result.array[i][j] = array[j][i];
        }
    }
    return result;
}

// takes as parameter a function which prototype looks like :
// double function(double x)
Matrix Matrix::applyFunction(double (*function)(double)) const{
    Matrix result(height, width);
    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            result.array[i][j] = (*function)(array[i][j]);
        }
    }
    return result;
}

// pretty print, taking into account the space between each element of the matrix
void Matrix::print(std::ostream &flux) const{
    int maxLength[width] = {};
    std::stringstream ss;

    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            ss << array[i][j];
            if(maxLength[j] < ss.str().size())
            {
                maxLength[j] = ss.str().size();
            }
            ss.str(std::string());
        }
    }

    for (int i=0; i<height; i++){
        for (int j=0; j<width; j++){
            flux << array[i][j];
            ss << array[i][j];
            for (int k=0 ; k<maxLength[j]-ss.str().size()+1 ; k++){
                flux << " ";
            }
            ss.str(std::string());
        }
        flux << std::endl;
    }
}

std::ostream& operator<<(std::ostream &flux, Matrix const &m){
    m.print(flux);
    return flux;
}
