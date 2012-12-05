#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_conjugate_gradient.h"
#include "vnl/vnl_cost_function.h"

class LeastSquaresFunctor : public vnl_cost_function
{
  public:
    LeastSquaresFunctor(vnl_matrix< double > &AInput,
	vnl_vector< double > &BInput) : vnl_cost_function(BInput.size())
    {
      SetA( AInput );
      SetB( BInput );
    }
    double f(const vnl_vector< double > & x)
    {
      vnl_vector< double > BEstimate = A * x;
      vnl_vector< double > BResidual = B - BEstimate;
      return BResidual.two_norm();
    }
    void gradf(const vnl_vector< double > & x, vnl_vector< double > &g)
    {
      g = A.transpose() * (A * x - B) * 2;
    }
    void SetB(vnl_vector< double > &BInput )
    {
      B = BInput;
    }
    void SetA(vnl_matrix< double > &AInput )
    {
      A = AInput;
    }
    vnl_matrix< double > GetA()
    {
      return A;
    }
    vnl_vector< double > GetB()
    {
      return B;
    }
  private:
    vnl_matrix< double > A;
    vnl_vector< double > B;
};

