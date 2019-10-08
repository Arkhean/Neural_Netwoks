/*
 * auteur : Julien Miens
 * date : 10/2019
 * description : test sur le réseau de neurones récursif
 */

#include "networks/rnn.hpp"
#include <time.h>

// arrondie à l'unité
double round(double x){
    if (x >= 0.5){
        return 1.0;
    }
    return 0.0;
}

// convertie une représentation binaire inversée sur 8 bits en entier
int binary_to_int(std::vector<double> v){
    int r = 0;
    int k = 1;
    for(double d : v){
        r += d * k;
        k *= 2;
    }
    return r;
}

// génère aléatoirement deux entiers < 128
// x contiendra la représentation binaire inversé de ces deux nombres (matrice à deux lignes)
// y contiendra la représentation binaire inversé de la somme (matrice ligne)
void generate_x_y(std::vector<std::vector<double>> &x, std::vector<std::vector<double>> &y){
    x.clear();
    y.clear();
    unsigned char a = rand() % 128;
    unsigned char b = rand() % 128;
    unsigned char c = a + b;    // résultat attendue
    for(int j = 0; j < 8; j++){
        x.push_back({(double)(a & 1), (double)(b & 1)});
        y.push_back({(double)(c & 1)});
        a >>= 1;
        b >>= 1;
        c >>= 1;
    }
}

// =============================================================================

// le test consiste à faire apprendre au réseau à faire une addition binaire
// plus précisement, le réseau doit apprendre à "faire la retenue" en posant
// l'opération
// l'entier (< 255) est représenté par un vecteur de taille 8 ne contenant que
// des 0 ou des 1 et inversé (le réseau lit les vecteurs de gauche à droite or
// on pose les opérations de droite à gauche...)

int main(int argc, char const *argv[]) {

    srand (time(NULL));
    RNN rnn(2, 16, 1, 0.1);
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;

    for(int i = 0; i < 15000; i++){
        generate_x_y(x,y);
        double e = rnn.learn(x, y);
        if (i % 1000 == 0){
            // on affiche l'erreur tous les 1000 itérations
            std::cout << e <<"\n";
        }
    }
    std::cout << '\n';

    // vérification sur un dernier test
    generate_x_y(x,y);
    Matrix Y = rnn.predict(x);
    Matrix C = y;
    Y = Y.applyFunction(round).transpose();
    C = C.transpose();
    Matrix X = Matrix(x).transpose();
    std::cout << binary_to_int(X.to_vector()[0]) << " + "
                << binary_to_int(X.to_vector()[1]);
    std::cout << " = " << binary_to_int(Y.to_vector()[0]) << '\n';
    std::cout << "got      : " << Y;
    std::cout << "expected : " << C;

    return 0;
}
