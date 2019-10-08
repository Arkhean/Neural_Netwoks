/*
 * auteur : Julien Miens
 * date : 10/2019
 * description : déclarations des classes Layer, Neuron_layer, Dropout_layer
 * ainsi que Network ; déclarations des fonctions d'apprentissage
 */

#ifndef NN_MOD_H
#define NN_MOD_H

#include "../utils/matrix.hpp"
#include "../utils/random.hpp"
#include "../utils/activation.hpp"

#include <typeinfo>
#include <algorithm>    // std::max ; sort


/* interface pour chaque niveau du réseau de Neurones */
class Layer{
    protected :
        int input_size;         // taille d'entrée
        int output_size;        // taille de sortie
        double alpha;           // taux d'apprentissage
        Matrix (*activ)(Matrix, bool);  // fonction d'activation associée
    public :
        virtual ~Layer(){}
        int get_input_size(){ return input_size; }
        int get_output_size(){ return output_size; }
        // prédit Y en fonction de l'entrée x (matrice ligne)
        virtual Matrix predict(Matrix x)=0;
        // réajuste les poids synaptiques du niveau en fonction du delta avec la
        // prédiction et de l'entrée
        // retourne le delta pour le niveau précédent
        virtual Matrix learn(Matrix x, Matrix delta)=0;
        // sert pour l'algorithme génétique
        // retourne un nouveau niveau dans lequel les synapses ont été mélangé
        // entre this et l2
        virtual Layer* fusion(Layer *l2)=0;
};

typedef double (*init_fun)(double);

/* Niveau de Neurones */
class Neuron_layer : public Layer{
    private :
        Matrix W;       // matrice des synapses
        Matrix B;       // matrice des biais
        init_fun random_f;  // fonction d'initialisation aléatoire des matrices
    public :
        ~Neuron_layer(){}
        // constructeur
        Neuron_layer(int input, int output, double alpha,
                                    Matrix (*activ)(Matrix, bool),
                                    init_fun=random_bis);
        Matrix predict(Matrix x);
        Matrix learn(Matrix x, Matrix delta);
        Layer * fusion(Layer *n2);
};

/* niveau "dropout", permet d'éviter l'overfitting */
class Dropout_layer : public Layer{
    private :
        double probability;     // probabilité de désactivation d'un neurone
    public :
        ~Dropout_layer(){}
        Dropout_layer(int size, double proba){
            probability = proba;
            input_size = size;  // la sortie est égale à la taille
            output_size = size;
        }
        // met chaque neurone avec une probabilité p à zero
        Matrix predict(Matrix x);
        // se contente de retransmettre delta
        Matrix learn(Matrix x, Matrix delta){ return delta; }
        // retourne une copie de lui même
        Layer * fusion(Layer *n2);
};

// =============================================================================

/* Classe contenant le réseau de neurones */
class Network{
    private :
        // successions de niveaux
        std::vector<Layer*> v;
    public :
        ~Network(){
            for(Layer *l : v){
                delete l;
            }
        }
        // ajoute un niveau à la fin de v
        // vérifie la correspondance des tailles sortie/entrée
        void add_layer(Layer* l);
        // envoie l'entrée à travers tous les niveaux successivement puis
        // retourne la prédiction
        Matrix predict(Matrix x);
        // utilise la backpropagation pour faire apprendre le réseau
        // fait prédire les niveaux un par un puis fait redescendre les deltas
        void learn(Matrix x, Matrix y);
        // génère un nouveau Network dont chaque niveau est la fusion de ceux
        // de this et n2
        // attention : les réseaux doivent avoir la même structure
        Network *fusion(Network *n2);
};

// =============================================================================

// on utilise un algorithme génétique pour trouver les poids du réseau
// on a besoin de structures intermédiaires
struct couple{
    Network *network;   // un réseau
    double error;       // son erreur vis-à-vis de l'entrée
};
bool tri(couple c1, couple c2);

Network * genetic_learning(double epsilon,      // erreur maximale acceptable
                int pop_size,           // nombre d'individus
                int max_gen,            // nombre maximale de générations
                int hidden_size,        // taille du niveau interne
                std::vector<std::vector<double>> x,     // tous les inputs
                std::vector<std::vector<double>> y,     // les labels correspondant
                // fonction d'estimation de l'erreur
                double (*calc_error)(Network*, std::vector<std::vector<double>>,
                        std::vector<std::vector<double>>),
                bool verbose,           // faire des print ?
                int nb_vue=10);         // affichage tous les nb_vue générations

Network * gradient_descent_learning(double epsilon,  // erreur maximale acceptable
                    int max_iterations,        // nombre maximal d'itérations
                    int hidden_size,           // taille du niveau interne
                    double learning_rate,      // taux d'apprentissage
                    std::vector<std::vector<double>> x, // entrée
                    std::vector<std::vector<double>> y, // sortie attendue
                    // fonction d'estimation de l'erreur
                    double (*calc_error)(Network*, std::vector<std::vector<double>>,
                                        std::vector<std::vector<double>>),
                    bool verbose,        // mode bavard
                    int nb_vue=10);      // affichage tous les nb_vues itérations


#endif
