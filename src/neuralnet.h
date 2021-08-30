/* 
Several types of activation functions and a basic class 
for a fully connected network that ALWAYS has a final layer
where each element activation function is tanh.
*/
#ifndef NEURALNET_H
#define NEURALNET_H

#include<fstream>

enum class Afunc : int {identity, sigmoid, tanh, relu, sthresh, gauss};

//These are defined here so that i/o routines can be centralized,
//rather than having each sim component do i/o. 
double identity(double x);
double sigmoid(double x);
double relu(double x);
double sthresh(double x);
double gauss(double x);

//This class currently assumes the last layer has width wf<=wi, while all
//intermediate layers have width wi.
class NeuralNet {
  public:
    //blank constructor
    NeuralNet(int wi, int wf, int d);
    //constructor from file
    NeuralNet(std::ifstream* ifile);
    void set_node(int i, int j, Afunc f);
    void set_bias(int i, double* b);
    void set_weights(int i, double* A);
    void mutate_params(double sdev);
    void mutate_neurons(double cprob);
    void evaluate(double* in, double* out);
    void store(std::ofstream* ofile);
    ~NeuralNet();

  private:
    int wi;
    int wf;
    int d;
    //all arrays have been flattened. 
    double* As;
    double* bs;
    //TODO: Figure out how to use an array of function pointers. 
    Afunc* nodes;
};

#endif //NEURALNET_H