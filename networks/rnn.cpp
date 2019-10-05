#include "rnn.hpp"

// =============================================================================
RNN::RNN(int input_size, int hidden_size, int output_size, double alpha){
    this->alpha = alpha;
    W1 = Matrix(input_size, hidden_size);
    W2 = Matrix(hidden_size, output_size);
    Wh = Matrix(hidden_size, hidden_size);
    W1 = W1.applyFunction(random_bis);
    W2 = W2.applyFunction(random_bis);
    Wh = Wh.applyFunction(random_bis);

    this->input_size = input_size;
    this->output_size = output_size;
    this->hidden_size = hidden_size;
}

double RNN::learn(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y){
    std::vector<Matrix> l2_deltas;
    std::vector<Matrix> l1_values;
    l1_values.push_back(Matrix(1, hidden_size));
    Matrix dW1(input_size, hidden_size);
    Matrix dW2(hidden_size, output_size);
    Matrix dWh(hidden_size, hidden_size);

    double error = 0.0;

    for(unsigned int i = 0; i < x.size(); i++){
        Matrix X = Matrix({x[i]});
        Matrix l1 = (X.dot(W1) + l1_values.back().dot(Wh)).applyFunction(sigmoid);
        Matrix l2 = (l1.dot(W2)).applyFunction(sigmoid);
        Matrix Y = Matrix({y[i]});
        Matrix l2_error = Y - l2;
        std::vector<double> v = l2_error.to_vector()[0];
        for(double d : v){
            error += fabs(d);
        }
        l2_deltas.push_back(l2_error * l2.applyFunction(sigmoidePrime_2));
        l1_values.push_back(l1);
    }

    Matrix future_l1_d = Matrix(1, hidden_size);
    for(unsigned int i = 0; i < x.size(); i++){
        Matrix l1 = l1_values[l1_values.size()-i-1];
        Matrix l1_prev = l1_values[l1_values.size()-i-2];
        Matrix l2_d = l2_deltas[l2_deltas.size()-i-1];
        Matrix l1_d = (future_l1_d.dot(Wh.transpose()) +
                    l2_d.dot(W2.transpose())) * l1.applyFunction(sigmoidePrime_2);

        Matrix X = Matrix({x[x.size()-i-1]});
        dW2 = dW2 + l1.transpose().dot(l2_d);
        dWh = dWh + l1_prev.transpose().dot(l1_d);
        dW1 = dW1 + X.transpose().dot(l1_d);
        future_l1_d = l1_d;
    }

    W1 = W1 + (dW1 * alpha);
    W2 = W2 + (dW2 * alpha);
    Wh = Wh + (dWh * alpha);
    return error;
}

std::vector<std::vector<double>> RNN::predict(std::vector<std::vector<double>> x){
    std::vector<Matrix> l1_values;
    l1_values.push_back(Matrix(1, hidden_size));
    std::vector<std::vector<double>> y;
    for(unsigned int i = 0; i < x.size(); i++){
        Matrix X = Matrix({x[i]});
        Matrix l1 = (X.dot(W1) + l1_values.back().dot(Wh)).applyFunction(sigmoid);
        Matrix l2 = l1.dot(W2).applyFunction(sigmoid);
        l1_values.push_back(l1);
        y.push_back(l2.to_vector()[0]);
    }
    return y;
}
