#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath> // exp
#include "matrix.hpp"

double sigmoid(double x);
double sigmoidePrime(double x); // true derivative
double sigmoidePrime_2(double x); // variant for RNN
double relu(double x);
double reluPrime(double x);
double tan_h(double x);
double tan_h_Prime(double x);
double id(double x);
double idPrime(double x);

// return derivative when b == true
Matrix id_m(Matrix x, bool b);
Matrix sigmoid_m(Matrix x, bool b);
Matrix sigmoid_m2(Matrix x, bool b); // variant for RNN
Matrix relu_m(Matrix x, bool b);
Matrix tan_h_m(Matrix x, bool b);
Matrix softmax(Matrix x, bool b);

#endif
