/*
 * auteur : Julien Miens
 * date : 10/2019
 * description : implémentation des classes de nn.hpp et des fonctions
 * d'apprentissage
 * voir le fichier nn.hpp pour les explications sur les fonctions
 */

#include "nn.hpp"

Neuron_layer::Neuron_layer(int input, int output, double alpha,
                                        Matrix (*activ)(Matrix, bool),
                                        init_fun random_f){
    this->input_size = input;
    this->output_size = output;
    this->alpha = alpha;
    this->activ = activ;
    this->random_f = random_f;
    W = Matrix(input, output);
    B = Matrix(1, output_size);

    W = W.applyFunction(random_f);
    B = B.applyFunction(random_f);
}

Matrix Neuron_layer::predict(Matrix x){
    return activ((x.dot(W) + B), false);
}

Matrix Neuron_layer::learn(Matrix x, Matrix delta){
    // compute gradient
    Matrix dB = delta * activ(x.dot(W)+B, true);
    Matrix dW = x.transpose().dot(dB);
    Matrix temp = W.transpose();
    // update weights
    W = W - (dW * alpha);
    B = B - (dB * alpha);
    return dB.dot(temp);
}

Layer *Neuron_layer::fusion(Layer *n2){
    Neuron_layer *n3 = new Neuron_layer(input_size, output_size, alpha, activ);
    Neuron_layer * n22 = static_cast<Neuron_layer*>(n2);
    for(int i = 0; i < output_size; i++){
        for(int j = 0; j < input_size; j++){
            double p = random(p);
            if (p < 0.49){ n3->W(j,i) = this->W(j,i); }
            else if (p < 0.98){ n3->W(j,i) = n22->W(j,i); }
            // else : laisser le gène aléatoire
        }
        double p = random(p);
        if (p < 0.49){ n3->B(0,i) = this->B(0,i); }
        else if (p < 0.98){ n3->B(0,i) = n22->B(0,i); }
    }
    return n3;
}

// =============================================================================

Matrix Dropout_layer::predict(Matrix x){
    Matrix y(x);
    for(int i = 0; i < x.get_rows(); i++){
        for(int j = 0; j < x.get_cols(); j++){
            double r = random(0);
            if (r < probability){
                y(i, j) = 0;
            }
        }
    }
    return y;
}

Layer * Dropout_layer::fusion(Layer *n2){
    return new Dropout_layer(input_size, probability);
}

// =============================================================================

void Network::add_layer(Layer* n){
    if (!v.empty() && v.back()->get_output_size() != n->get_input_size()){
        std::cout << "error ! dim not aligned ! (" << v.back()->get_output_size()
                    << " != " << n->get_input_size() << ")\n";
    }
    else{
        v.push_back(n);
    }
}

Matrix Network::predict(Matrix x){
    Matrix Y(x);
    for(Layer* n : v){
        Y = n->predict(Y);
    }
    return Y;
}

void Network::learn(Matrix input, Matrix expected){
    Matrix x_s[v.size()+1];
    x_s[0] = input;
    for(unsigned int i = 1; i < v.size()+1; i++){
        x_s[i] = v[i-1]->predict(x_s[i-1]);
    }
    Matrix delta = x_s[v.size()] - Matrix({expected});
    for(unsigned int i = v.size(); i > 0; i--){
        delta = v[i-1]->learn(x_s[i-1], delta);
    }
}

Network *Network::fusion(Network *n2){
    Network *n3 = new Network;
    for(unsigned int i = 0; i < v.size(); i++){
        n3->add_layer(v[i]->fusion(n2->v[i]));
    }
    return n3;
}

// =============================================================================

bool tri(couple c1, couple c2){
    return c1.error < c2.error;
}

// retourne une matrice de même forme que x mais dont les coefficients sont tous
// nuls sont le plus grand de x
Matrix max(Matrix x){
    Matrix y(x);
    int x_m = 0;
    int y_m = 0;
    for(int i = 0; i < x.get_rows(); i++){
        for(int j = 0; j < x.get_cols(); j++){
            if (x(i,j) > x(x_m, y_m)){
                x_m = i;
                y_m = j;
            }
            y(i,j) = 0;
        }
    }
    y(x_m, y_m) = 1;
    return y;
}

Network *genetic_learning(double epsilon, int pop_size, int max_gen,
            int hidden_size,
            std::vector<std::vector<double>> x,
            std::vector<std::vector<double>> y,
            double (*calc_error)(Network*, std::vector<std::vector<double>>,
                std::vector<std::vector<double>>),
            bool verbose,
            int nb_vue){

    auto init_fun_rand = [](double x) -> double { return random(x, 10.0); };

    int elite = pop_size/10;
    int new_pop = pop_size/10;

    std::vector<couple> population;
    for(int i = 0; i < pop_size; i++){
        Network *n = new Network;
        n->add_layer(new Neuron_layer(x[0].size(), hidden_size, 0.1, sigmoid_m,
                                                                init_fun_rand));
        n->add_layer(new Neuron_layer(hidden_size, y[0].size(), 0.1, sigmoid_m,
                                                                init_fun_rand));
        couple c = {n, 0.0};
        population.push_back(c);
    }

    int gen;
    for(gen = 0; gen < max_gen; gen++){
        for(int i = 0; i < pop_size; i++){
            population[i].error = calc_error(population[i].network, x, y);
        }
        sort(population.begin(), population.end(), tri);
        // si le meilleur est bon, on peut arrêter
        if (population[0].error < epsilon){
            break;
        }
        if (verbose && (gen % nb_vue) == 0){
            std::cout << "generations : " << gen;
            std::cout << " ; error = " << population[0].error << '\n';
            int k = rand() % x.size();
            std::cout << "expected : " << y[k]
                      << " got     : " << max(population[0].network->predict(x[k])) << "\n";
        }
        std::vector<couple> new_generation;
         // on conserve les meilleurs (élitisme)
        for(int i = 0; i < elite; i++){
            new_generation.push_back(population[i]);
        }
        // on fait se reproduire les 30% meilleurs
        for(int i = 0; i < pop_size-elite-new_pop; i++){
            int r = random(0, pop_size/3);
            Network *parent1 = population[r].network;
            r = random(0, pop_size/3);
            Network *parent2 = population[r].network;
            Network *offspring = parent1->fusion(parent2);
            couple c = {offspring, 0.0};
            new_generation.push_back(c);
        }
        // on ajoute de la diversité avec de nouveaux aléatoires
        for(int i = 0; i < new_pop; i++){
            Network * n = new Network;
            n->add_layer(new Neuron_layer(x[0].size(), hidden_size, 0.1,
                                                                    sigmoid_m));
            n->add_layer(new Neuron_layer(hidden_size, y[0].size(), 0.1,
                                                                    sigmoid_m));
            couple c = {n, 0.0};
            new_generation.push_back(c);
        }
        for(int i = elite; i < pop_size; i++){
            delete population[i].network;
        }
        population = new_generation;
    }
    if (verbose){
        std::cout << "final error : " << population[0].error << '\n';
        std::cout << "number of generations : " << gen << '\n';

        for(int j = 0; j < 5; j++){
            int k = rand() % x.size();
            std::cout << "expected : " << y[k] << " got : "
            << population[j].network->predict(x[k]) << "\n";
        }
    }
    for(int i = 1; i < pop_size; i++){
        delete population[i].network;
    }
    return population[0].network;
}

// =============================================================================

Network * gradient_descent_learning(double epsilon, int max_iterations,
            int hidden_size,
            double learning_rate, std::vector<std::vector<double>> x,
            std::vector<std::vector<double>> y,
            double (*calc_error)(Network*, std::vector<std::vector<double>>,
                std::vector<std::vector<double>>),
                bool verbose, int nb_vue){
    Network * nn_test = new Network;
    nn_test->add_layer(new Neuron_layer(x[0].size(), hidden_size, learning_rate,
                                                                    sigmoid_m));
    nn_test->add_layer(new Neuron_layer(hidden_size, y[0].size(), learning_rate,
                                                                    sigmoid_m));

    double error;
    int iterations = 0;
    do{
        iterations++;
        if (iterations > max_iterations){
            break;
        }
        for(unsigned int j = 0; j < x.size(); j++){
            // make a prediction and autocorrect with gradient descent
            nn_test->learn(x[j], y[j]);
        }
        error = calc_error(nn_test, x, y);
        if (verbose && (iterations % nb_vue) == 0){
            std::cout << "iterations : " << iterations;
            std::cout << " ; error = " << error << '\n';
        }
    }while(error >= epsilon);
    if (verbose){
        std::cout << "final error : " << error << '\n';
        std::cout << "number of iterations : " << iterations << '\n';

        for(int j = 0; j < 5; j++){
            int k = rand() % x.size();
            std::cout << "expected : " << y[k] << " got : "
            << nn_test->predict(x[k]) << "\n";
        }
    }
    return nn_test;
}
