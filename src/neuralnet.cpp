#include<cmath>
#include<cstdlib>
#include<random>
#include "neuralnet.h"

double identity(double x){
  return x;
}

double sigmoid(double x){
  return 1.0/(1.0+std::exp(-1.0*x));
}

double relu(double x){
  if(x>0){
    return x;
  }else{
    return 0;
  }
}

double sthresh(double x){
  double sgn;
  if(x>=0){
    sgn = 1.0;
  } else {
    sgn = -1.0;
  }
  return sgn*relu(std::abs(x)-1.0);
}

double gauss(double x){
  return std::exp(x*x);
}

NeuralNet::NeuralNet(int wi, int wf, int d){
  wi = wi;
  wf = wf;
  d = d;
  As = (double*) malloc(sizeof(double)*((d-1)*wi*wi + wi*wf));
  bs = (double*) malloc(sizeof(double)*(wi*(d-1)+wf));
  nodes = (Afunc*) malloc(sizeof(Afunc)*(wi*(d-1)+wf));
  for(int i=0; i<(d-1)*wi*wi + wi*wf; ++i){
    As[i] = 0.0;
  }
  for(int i=0; i<wi*(d-1)+wf; ++i){
    bs[i] = 0.0;
  }
  for(int i=0; i<wi*(d-1)+wf; ++i){
    nodes[i] = Afunc::tanh;
  }
}

NeuralNet::NeuralNet(std::ifstream* ifile){

}

void NeuralNet::set_node(int i, int j, Afunc f){
  int idx = j + wi*i;
  nodes[idx] = f;
}

void NeuralNet::set_bias(int i, double* b){
  int idx;
  if(i<d-1){
    for(int j=0; j<wi; ++j){
      idx = j+wi*i;
      bs[idx] = b[j];
    }
  }else{
    for(int j=0; j<wf; ++j){
      idx = j + wi*(d-1);
      bs[idx] = b[j];
    }
  }
}

void NeuralNet::set_weights(int i, double* A){
  int idx1;
  int idx2; 
  //i: which layer
  //j: which row of layer i
  //k: which column of row j
  if(i<d-1){
    for(int j=0; j<wi; ++j){
      for(int k=0; k<wi; ++k){
        idx1 = k + wi*(j+wi*i);
        idx2 = k + wi*j;
        As[idx1] = A[idx2];
      }
    }
  } else{
    for(int j=0; j<wf; ++j){
      for(int k=0; k<wi; ++k){
        idx1 = k + wi*(j+wi*i);
        idx2 = k + wi*j;
        As[idx1] = A[idx2];
      }
    }
  }
}

void NeuralNet::mutate_params(double sdev){
  std::random_device rd{};
  std::mt19937 gen{rd()};
  //mutate weights
  for(int i=0; i<wi*((d-1)*wi+wf); ++i){
    std::normal_distribution<double> nvar(As[i], sdev);
    As[i] = nvar(gen);
  }
  //mutate biases
  for(int i=0; i<wi*(d-1)+wf; ++i){
    std::normal_distribution<double> nvar(bs[i], sdev);
    bs[i] = nvar(gen);
  }
}

//cprob: probability of keeping the same neuron, 
//otherwise choose uniformly amongst other options
//This function is probably very dangerous --mutating neurons is going to be violent.
void NeuralNet::mutate_neurons(double cprob){
  std::random_device rd{};
  std::mt19937 gen(rd());
  double w1 = cprob;
  double w2 = (1.0-cprob)/5.0;
  std::uniform_int_distribution<> dis(0, 5);
  //last layer could be tanh only. tbd.
  for(int i=0; i<(d-1)*wi+wf; ++i){
    //sloppy way to code this, revise later if time allows.
    //Find the type of params in std::discrete_distribution to make loop.
    if(nodes[i]==Afunc::identity){
      std::discrete_distribution<> d({w1, w2, w2, w2, w2, w2});
      nodes[i] = static_cast<Afunc>(d(gen));
    }else if(nodes[i]==Afunc::sigmoid){
      std::discrete_distribution<> d({w2, w1, w2, w2, w2, w2});
      nodes[i] = static_cast<Afunc>(d(gen));
    }else if(nodes[i]==Afunc::tanh){
      std::discrete_distribution<> d({w2, w2, w1, w2, w2, w2});
      nodes[i] = static_cast<Afunc>(d(gen));
    }else if(nodes[i]==Afunc::relu){
      std::discrete_distribution<> d({w2, w2, w2, w1, w2, w2});
      nodes[i] = static_cast<Afunc>(d(gen));
    }else if(nodes[i]==Afunc::sthresh){
      std::discrete_distribution<> d({w2, w2, w2, w2, w1, w2});
      nodes[i] = static_cast<Afunc>(d(gen));
    } else{
      std::discrete_distribution<> d({w2, w2, w2, w2, w2, w1});
      nodes[i] = static_cast<Afunc>(d(gen));
    }
  }
}

void NeuralNet::evaluate(double* in, double* out){
  //note: there appears to be a way to not evaluate these at
  //each call of the inner most loop where they are used. 
  //Check the index arithmetic; very minor savings that stack as loop grows.
  int bidx;
  int aidx;
  int nidx;
  int jlim;
  double* temp1 = (double*) malloc(sizeof(double)*wi);
  double* temp2 = (double*) malloc(sizeof(double)*wi);
  //Initialize out values
  for(int i=0; i<wi; ++i){
    temp1[i] = in[i];
  }
  //loop over layers
  for(int i=0; i<d; ++i){
    jlim = i<d-1 ? wi : wf;
    //loop over rows of layer i
    for(int j=0; j<jlim; ++j){
      bidx = j+wi*i;
      temp2[j] = bs[bidx];
      //loop over columns of row j
      for(int k=0; k<wi; ++j){
        aidx = k + wi*(j+wi*i);
        temp2[j] += As[aidx]*temp1[k];
      }
    }
    //for cache locality, use activation functions after Ax+b op.
    for(int j=0; j<jlim; ++j){
      nidx = j+wi*i;
      //this is where the benefit of using array of func pointers appears.
      if(nodes[nidx]==Afunc::identity){
        temp1[j] = identity(temp2[j]);
      }else if(nodes[nidx]==Afunc::sigmoid){
        temp1[j] = sigmoid(temp2[j]);
      }else if(nodes[nidx]==Afunc::tanh){
        temp1[j] = tanh(temp2[j]);
      }else if(nodes[nidx]==Afunc::relu){
        temp1[j] = relu(temp2[j]);
      }else if(nodes[nidx]==Afunc::sthresh){
        temp1[j] = sthresh(temp2[j]);
      }else{
        temp1[j] = gauss(temp2[j]);
      }
    }
  }
  for(int i=0; i<wf; ++i){
    out[i] = temp1[i];
  }
  std::free(temp1);
  std::free(temp2);
}

NeuralNet::~NeuralNet(){
  std::free(As);
  std::free(bs);
  std::free(nodes);
}


