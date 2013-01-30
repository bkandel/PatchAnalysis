#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_conjugate_gradient.h"
#include "vnl/vnl_cost_function.h"
#include "vnl/algo/vnl_amoeba.h"
#include "vnl/algo/vnl_lbfgs.h"


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
  return VectorAsSpatialImage;
};
