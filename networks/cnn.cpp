/*
 * TRAVAIL EN COURS...
 */

#include "cnn.hpp"

Convolution_layer::Convolution_layer(int nb_kernels, int kernel_size, double alpha){
    this->nb_kernels = nb_kernels;
    this->kernel_size = kernel_size;
    for(int i = 0; i < nb_kernels; i++){
        Matrix K(kernel_size, kernel_size);
        K = K.applyFunction(random_bis);
        kernels.push_back(K);
    }
}

std::vector<Matrix> Convolution_layer::predict(std::vector<Matrix> x){
    std::vector<Matrix> res;
    for(Matrix H : x){
        for(Matrix K : kernels){
            res.push_back(H.convolution(K));
        }
    }
    return res;
}

std::vector<Matrix> learn(std::vector<Matrix> x, std::vector<Matrix> delta){

}

// =============================================================================

std::vector<Matrix> Activation_layer::predict(std::vector<Matrix> x){
    std::vector<Matrix> v;
    for(Matrix M : x){
        v.push_back(M.applyFunction(relu));
    }
    return v;
}

// =============================================================================

std::vector<Matrix> Pooling_layer::predict(std::vector<Matrix> x){
    std::vector<Matrix> v;
    for(Matrix M : x){
        Matrix r(M.get_rows()-pool_size+1, M.get_cols()-pool_size+1);
        for(int i = 0; i < r.get_rows(); i++){
            for(int j = 0; j < r.get_cols(); j++){
                for(int k = 0; k < pool_size; k++){
                    for(int l = 0; l < pool_size; l++){
                        r(i,j) = std::max(r(i,j), M(i+k,j+l));
                    }
                }
            }
        }
        v.push_back(r);
    }
    return v;
}


// =============================================================================

std::vector<Matrix> Dropout_layer::predict(std::vector<Matrix> x){
    std::vector<Matrix> v;
    for(Matrix H : x){
        Matrix y(H);
        for(int i = 0; i < H.get_rows(); i++){
            for(int j = 0; j < H.get_cols(); j++){
                if (random(0) < probability){
                    y(i, j) = 0;
                }
            }
        }
        v.push_back(y);
    }
    return v;
}
