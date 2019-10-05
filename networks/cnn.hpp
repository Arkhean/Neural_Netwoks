#ifndef CNN_H
#define CNN_H

#include "../utils/matrix.hpp"
#include "../utils/random.hpp"
#include "../utils/activation.hpp"

class Layer{
    virtual std::vector<Matrix> predict(std::vector<Matrix> x);
    virtual std::vector<Matrix> learn(std::vector<Matrix> x, std::vector<Matrix> delta);
};

class Activation_layer : public Layer{
    std::vector<Matrix> predict(std::vector<Matrix> x);
    std::vector<Matrix> learn(std::vector<Matrix> x, std::vector<Matrix> delta);
};

class Pooling_layer : public Layer{
    private :
        int pool_size;
    public :
        Pooling_layer(int size) : pool_size(size){}
        std::vector<Matrix> predict(std::vector<Matrix>x );
        std::vector<Matrix> learn(std::vector<Matrix> x, std::vector<Matrix> delta);
};

class Dropout_layer : public Layer{
    private :
        int probability;
    public :
        Dropout_layer(int proba) : probability(proba){};
        // met chaque neurone avec une probabilité p à zero
        // se contente de retransmettre delta
        std::vector<Matrix> predict(std::vector<Matrix> x);
        std::vector<Matrix> learn(std::vector<Matrix> x, std::vector<Matrix> delta){ return delta; }
};

class Convolution_layer : public Layer{
    private :
        int nb_kernels;
        int kernel_size;
        std::vector<Matrix> kernels;
        // Matrix B; // eventuellement un biais ...
    public :
        Convolution_layer(int nb_kernels, int kernel_size, double alpha);
        std::vector<Matrix> predict(std::vector<Matrix> x);
        std::vector<Matrix> learn(std::vector<Matrix> x, std::vector<Matrix> delta);
};

// =============================================================================

class CNN{

};

#endif
