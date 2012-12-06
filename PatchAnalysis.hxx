#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_conjugate_gradient.h"
#include "vnl/vnl_cost_function.h"
#include "vnl/algo/vnl_amoeba.h"
#include "vnl/algo/vnl_lbfgs.h"

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

template< class InputImageType, class InputImage, class InputPixelType >
typename InputImageType::Pointer ConvertVectorToSpatialImage( vnl_vector< InputPixelType > &Vector, 
      typename InputImage::Pointer Mask)
{
  typename InputImageType::Pointer VectorAsSpatialImage = InputImageType::New(); 
  VectorAsSpatialImage->SetOrigin(Mask->GetOrigin() );
  VectorAsSpatialImage->SetSpacing( Mask->GetSpacing() ); 
  VectorAsSpatialImage->SetRegions( Mask->GetLargestPossibleRegion() ); 
  VectorAsSpatialImage->SetDirection( Mask-> GetDirection() ); 
  VectorAsSpatialImage->Allocate(); 
  VectorAsSpatialImage->FillBuffer( itk::NumericTraits<InputPixelType>::Zero ); 
  unsigned long VectorIndex = 0; 
  typedef itk::ImageRegionIteratorWithIndex<InputImage> IteratorType; 
  IteratorType MaskIterator( Mask, Mask->GetLargestPossibleRegion() );
  for( MaskIterator.GoToBegin(); !MaskIterator.IsAtEnd(); ++MaskIterator)
  {
    if( MaskIterator.Get() >= 0.5 )
    {
      InputPixelType Value = 0.0; 
      if( VectorIndex < Vector.size() )
      {
	Value = Vector(VectorIndex); 
      }
      else
      {
	std::cout << "Size of mask does not match size of vector to be written!" << std::endl;
	std::cout << "Exiting." << std::endl;
	std::exception(); 
      }
      VectorAsSpatialImage->SetPixel(MaskIterator.GetIndex(), Value); 
      ++VectorIndex; 
    }
    else
    {
      MaskIterator.Set( 0 ); 
    }
  }
};
