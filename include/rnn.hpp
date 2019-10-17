#ifndef RNN_H
#define RNN_H

#include "matrix.hpp"
#include "random.hpp"
#include "activation.hpp"

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
        // retourne une séquence de même longueur que l'entrée
        // chaque élément ayant été prédit par le réseau en connaissance des
        // précédents
        std::vector<std::vector<double>> predict(std::vector<std::vector<double>> x);
        // fait apprendre une séquence
        // action à répéter beaucoup pour atteindre la convergence
        double learn(std::vector<std::vector<double>> x,
                            std::vector<std::vector<double>> y);
};

// TODO : fonction pour apprendre une liste de séquences de vecteurs

#endif
