#include "../include/activation.hpp"

double sigmoid(double x){
    return 1/(1+exp(-x));
}

double sigmoidePrime(double x){
    return exp(-x)/(pow(1+exp(-x), 2));
}

double sigmoidePrime_2(double x){
    return x*(1-x);
}

double relu(double x){
    if (x > 0){
        return x;
    }
    return 0;
}

double reluPrime(double x){
    if (x > 0){
        return 1;
    }
    return 0;
}

double tan_h(double x){
    return tanh(x);
}

double tan_h_Prime(double x){
    return 1 - pow(x, 2);
}

double id(double x){return x;}
double idPrime(double x){return 1;}

// =============================================================================
Matrix id_m(Matrix x, bool b){
    if (b){
        return x.applyFunction(idPrime);
    }
    return x;
}

Matrix sigmoid_m(Matrix x, bool b){
    if (b){
        return x.applyFunction(sigmoidePrime);
    }
    return x.applyFunction(sigmoid);
}

Matrix sigmoide_m2(Matrix x, bool b){
    if (b){
        return x.applyFunction(sigmoidePrime_2);
    }
    return x.applyFunction(sigmoid);
}

Matrix relu_m(Matrix x, bool b){
    if (b){
        return x.applyFunction(reluPrime);
    }
    return x.applyFunction(relu);
}


Matrix tan_h_m(Matrix x, bool b){
    if (b){
        return x.applyFunction(tan_h_Prime);
    }
    return x.applyFunction(tan_h);
}

Matrix softmax(Matrix x, bool b){
    if (!b){
        double s = 0.0;
        for(int i = 0; i < x.get_cols(); i++){
            s += exp(x(0,i));
        }
        Matrix y(x);
        for(int i = 0; i < x.get_cols(); i++){
            y(0,i) = x(0,i) / s;
        }
        return y;
    }
    double s = 0.0;
    for(int i = 0; i < x.get_cols(); i++){
        s += exp(x(0,i));
    }
    Matrix y(x);
    for(int i = 0; i < x.get_cols(); i++){
        y(0,i) = (x(0,i) / s) * (1 - (x(0,i) / s));
    }
    return y;
}
