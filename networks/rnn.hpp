#ifndef RNN_H
#define RNN_H

#include "../utils/matrix.hpp"
#include "../utils/random.hpp"
#include "../utils/activation.hpp"

class RNN{
    private :
        double alpha;   // learning rate
        Matrix W1;      // synapse 1
        Matrix W2;      // synapse 2
        Matrix Wh;      // synapse hidden
        int input_size;
        int output_size;
        int hidden_size;
    public :
        RNN(int input_size, int hidden_size, int output_size, double alpha);
        std::vector<std::vector<double>> predict(std::vector<std::vector<double>> x);
        // pour faire apprendre une séquence de vecteurs
        double learn(std::vector<std::vector<double>> x,
                            std::vector<std::vector<double>> y);
};

#endif
