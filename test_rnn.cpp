#include "networks/rnn.hpp"
#include <time.h>

double round(double x){
    if (x >= 0.5){
        return 1.0;
    }
    return 0.0;
}

int binary_to_int(std::vector<double> v){
    int r = 0;
    int k = 1;
    for(double d : v){
        r += d * k;
        k *= 2;
    }
    return r;
}

int main(int argc, char const *argv[]) {
    RNN rnn(2, 16, 1, 0.1);
    srand (time(NULL));

    for(int i = 0; i < 20000; i++){
        unsigned char a = rand() % 128;
        unsigned char b = rand() % 128;
        unsigned char c = a + b;
        std::vector<std::vector<double>> x;
        std::vector<std::vector<double>> y;
        for(int j = 0; j < 8; j++){
            x.push_back({(double)(a & 1), (double)(b & 1)});
            y.push_back({(double)(c & 1)});
            a >>= 1;
            b >>= 1;
            c >>= 1;
        }
        double e = rnn.learn(x, y);
        if (i % 1000 == 0){
            std::cout << e <<" ";
        }
    }
    std::cout << '\n';

    unsigned char a = rand() % 128;
    unsigned char b = rand() % 128;
    unsigned char c = a + b;
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;
    std::cout << (unsigned int)a << " + " << (unsigned int)b;
    for(int j = 0; j < 8; j++){
        x.push_back({(double)(a & 1), (double)(b & 1)});
        y.push_back({(double)(c & 1)});
        a >>= 1;
        b >>= 1;
        c >>= 1;
    }
    Matrix Y = rnn.predict(x);
    Matrix C = y;
    Y = Y.applyFunction(round).transpose();
    C = C.transpose();
    std::cout << " = " << binary_to_int(Y.to_vector()[0]) << '\n';
    std::cout << "got      : " << Y;
    std::cout << "expected : " << C;

    return 0;
}
