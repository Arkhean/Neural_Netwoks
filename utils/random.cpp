#include "../include/random.hpp"
#include <iostream>     // rand

// need srand before ...

// return double between 0.0 and 1.0
double random(double x){
    return (double)(rand() % 10000 + 1)/10000;
}

double random(double x, double range){
    return range * (2 * random(x) - 1);
}

// return double between -1.0 and 1.0
double random_bis(double x){
    return 2*random(x)-1;
}

// return int between start and end
int random(int start, int end){
    int range = (end-start)+1;
    int random_int = start+(rand()%range);
    return random_int;
}
