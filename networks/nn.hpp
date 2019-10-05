#ifndef NN_MOD_H
#define NN_MOD_H

#include "../utils/matrix.hpp"
#include "../utils/random.hpp"
#include "../utils/activation.hpp"

#include <typeinfo>
#include <algorithm>    // std::max ; sort

class Layer{
    protected :
        int input_size;
        int output_size;
        double alpha;
        Matrix (*activ)(Matrix, bool);
    public :
        virtual ~Layer(){}
        int get_input_size(){ return input_size; }
        int get_output_size(){ return output_size; }
        virtual Matrix predict(Matrix x)=0;
        virtual Matrix learn(Matrix x, Matrix delta)=0;
        virtual Layer* fusion(Layer *l2)=0;
};

typedef double (*init_fun)(double);

class Neuron_layer : public Layer{
    private :
        Matrix W;
        Matrix B;
        init_fun random_f;
    public :
        ~Neuron_layer(){}
        Neuron_layer(int input, int output, double alpha,
                                    Matrix (*activ)(Matrix, bool),
                                    init_fun=random_bis);
        Matrix predict(Matrix x);
        Matrix learn(Matrix x, Matrix delta);
        Layer * fusion(Layer *n2);
};

class Dropout_layer : public Layer{
    private :
        double probability;
    public :
        ~Dropout_layer(){}
        Dropout_layer(int size, double proba){
            probability = proba;
            input_size = size;
            output_size = size;
        }
        // met chaque neurone avec une probabilité p à zero
        // se contente de retransmettre delta
        Matrix predict(Matrix x);
        Matrix learn(Matrix x, Matrix delta){ return delta; }
        Layer * fusion(Layer *n2);
};

// =============================================================================

class Network{
    private :
        std::vector<Layer*> v;
    public :
        ~Network(){
            for(Layer *l : v){
                delete l;
            }
        }
        void add_layer(Layer* l);
        Matrix predict(Matrix x);
        // utilise la backpropagation pour faire apprendre le réseau
        // fait prédire les niveaux un par un puis fait redescendre les deltas
        void learn(Matrix x, Matrix y);
        Network *fusion(Network *n2);
};

// =============================================================================

// on utilise un algorithme génétique pour trouver les poids du réseau
// on a besoin de structures intermédiaires
struct couple{
    Network *network;
    double error;
};
bool tri(couple c1, couple c2);

Network * genetic_learning(double epsilon,         // erreur maximale acceptable
                    int pop_size,           // nombre d'individus
                    int max_gen,            // nombre maximale de générations
                    int hidden_size,        // taille du niveau interne
                    std::vector<std::vector<double>> x,     // tous les inputs
                    std::vector<std::vector<double>> y,     // les labels correspondant
                    double (*calc_error)(Network*, std::vector<std::vector<double>>,  // fonction d'estimation de l'erreur
                        std::vector<std::vector<double>>),
                    bool verbose,       // faire des print ?
                    int nb_vue=10);

Network * gradient_descent_learning(double epsilon,
                            int max_iterations,
                            int hidden_size,
                            double learning_rate,
                            std::vector<std::vector<double>> x,
                            std::vector<std::vector<double>> y,
                            double (*calc_error)(Network*, std::vector<std::vector<double>>,
                                                        std::vector<std::vector<double>>),
                            bool verbose,
                            int nb_vue=10);


#endif
