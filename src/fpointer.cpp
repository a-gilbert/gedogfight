#include <iostream> 
#include <cstdlib>

//type of function taking in double returning double;
typedef double (*dfpointer)(double);

double foo(double x)
{
  return 2.0*x;
}

int main()
{
  //temporary pointer
  double (*tpointer)(double);
  //2d array of functions? 
  dfpointer** farray = malloc(3*sizeof(*dfpointer));

  farray = malloc(3*sizeof(tpointer));
}