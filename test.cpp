/*
 * auteur : Julien Miens
 * date : 10/2019
 * description : tests sur les réseaux de neurones
 * contient les fonctions d'import des données pour IRIS et MNIST
 */

#include <bits/stdc++.h>    // sort vector
#include <algorithm>        // std::max

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

#include "networks/nn.hpp"

// calcul de l'erreur comme somme des carrés des différences prédit/réel
double calculate_error(Network *network, std::vector<std::vector<double>> x,
    std::vector<std::vector<double>> y)
{
    double e = 0.0;
    for(unsigned int j = 0; j < x.size(); j++){
        e += pow(fabs(network->predict(x[j])(0,0) - y[j][0]), 2);
    }
    return e;
}

// =============================================================================

int load_mnist(std::vector<std::vector<double>> & x,
                std::vector<std::vector<double>> & y, int nb){

    x.clear();
    y.clear();

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<double>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, double, uint8_t>("data");

    binarize_dataset(dataset);
    std::vector<std::vector<double>> images(dataset.training_images.begin(),
                                            dataset.training_images.begin()+nb);
    std::vector<uint8_t> labels(dataset.training_labels.begin(),
                                            dataset.training_labels.begin()+nb);

    x = images;
    for(double b : labels){
        std::vector<double> v = {0,0,0,0,0,0,0,0,0,0};
        v[b] = 1;
        y.push_back(v);
    }
    return 0;
}

int load_iris(std::vector<std::vector<double>> & x,
                std::vector<std::vector<double>> & y){
    FILE * pFile;
    x.clear();
    y.clear();
    pFile = fopen ("data/iris.txt" , "r");
    if (pFile == NULL){
        perror ("Error opening file");
        return -1;
    }
    else{
        char buffer[100];
        fgets(buffer, 100, pFile);  // sauter la première ligne
        while ( ! feof (pFile) ) {
            double a,b,c,d;
            char buffer[20];
            fscanf(pFile, "%lf,%lf,%lf,%lf,%s", &a, &b, &c, &d, buffer);
            x.push_back({a,b,c,d});
            std::vector<double> tmp = {0,0,0};
            if (strcmp(buffer, "setosa") == 0){
                tmp[0] = 1;
            }
            else if (strcmp(buffer, "versicolor") == 0){
                tmp[1] = 1;
            }
            else{
                tmp[2] = 1;
            }
            y.push_back(tmp);
        }
        fclose (pFile);
        return 0;
    }
}

// =============================================================================

// TODO : cross validation

int main(int argc, char * argv[]){
    srand(time(NULL));

    // premier test : apprendre le ou exclusif
    std::vector<std::vector<double>> x = {{0,0}, {0,1}, {1,0}, {1,1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

    // d'abord avec la méthode de descente graduelle
    std::cout << "Résolution OU_EXCLUSIF utilisant la descente graduelle" << '\n';
    Network * n = gradient_descent_learning(1e-4, 100000, 4, 0.5, x, y,
                                                    calculate_error, true);
    delete n;
    std::cout << "===================================================" << '\n';

    // ensuite avec l'algorithmique génétique
    std::cout << "Résolution OU_EXCLUSIF utilisant un algorithme génétique" << '\n';
    n = genetic_learning(1e-4, 1000, 1000, 4, x, y,
                                                calculate_error, true);
    delete n;
    std::cout << "===================================================" << '\n';

    // troisième test : apprentissage de classification sur la base de données
    // IRIS : 3 catégories de fleurs selon les dimensions des pétales/sépales
    // traitement un peu plus long
    std::cout << "Résolution IRIS utilisant la descente graduelle" << '\n';
    if (load_iris(x, y) != -1){
        n = gradient_descent_learning(1e-3, 10000, 6, 0.1, x, y,
                                                        calculate_error, true);
        delete n;
    }
    std::cout << "===================================================" << '\n';

    // quatrième test : reconnaissance des chiffres de MNIST
    // apprentissage assez long (+ 1h...)
    std::cout << "Résolution MNIST utilisant la descente graduelle" << '\n';
    if (load_mnist(x, y, 1000) != -1){
        n = gradient_descent_learning(1e-2, 1000, 32, 0.1, x, y,
                                                calculate_error, true, 1000);
        delete n;
    }
    return 0;
}
