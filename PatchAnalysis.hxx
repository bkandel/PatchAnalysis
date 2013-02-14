#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <itkImageRegionIterator.h>
#include "itkNeighborhoodIterator.h"

template <class TImage>
bool IsInside( typename TImage::Pointer input, typename TImage::IndexType index )
{
  /** FIXME - should use StartIndex - */
  typedef TImage ImageType;
  enum { ImageDimension = ImageType::ImageDimension };
  bool isinside = true;
  for( unsigned int i = 0; i < ImageDimension; i++ )
    {
    float shifted = index[i];
    if( shifted < 0 || shifted >  input->GetLargestPossibleRegion().GetSize()[i] - 1  )
      {
      isinside = false;
      }
    }
  return isinside;
}

template< class InputImageType,  class InputPixelType > 
typename InputImageType::Pointer GenerateMaskImageFromPatch( 
    std::vector< unsigned int > &IndicesWithinSphere, 
    const unsigned int &RadiusOfPatch, 
    const unsigned int &Dimension ) 
{
  int NumberOfPaddingVoxels = 2;  
  int SizeOfImage = 2 * RadiusOfPatch  + 2 *  NumberOfPaddingVoxels + 1; 
  typename InputImageType::Pointer MaskImage = InputImageType::New(); 
  typename InputImageType::IndexType   start;
  typename InputImageType::IndexType   BeginningOfSphereRegion; 
  typename InputImageType::SizeType    SizeOfSphereRegion; 
  typename InputImageType::SizeType    size;
  typename InputImageType::SpacingType spacing; 
  typename InputImageType::PointType   OriginPoint; 
  typename InputImageType::IndexType   OriginIndex;

  for( unsigned int dd = 0; dd < Dimension; dd++)
  {
    start[ dd ]   = 0; 
    size[ dd ]    = SizeOfImage; 
    spacing[ dd ] = 1.0; 
    OriginPoint[ dd ] = OriginIndex[ dd ]  = 0.0; 
    BeginningOfSphereRegion[ dd ] = NumberOfPaddingVoxels + RadiusOfPatch; // one for each side--this is correct
    SizeOfSphereRegion[ dd ] = RadiusOfPatch * 2 + 1; 
  }
  typename InputImageType::RegionType region; 
  region.SetSize( size ); 
  region.SetIndex( OriginIndex ); 
  MaskImage->SetRegions( region ); 
  MaskImage->Allocate( ); 
  MaskImage->SetSpacing( spacing ); 
  MaskImage->SetOrigin( OriginPoint );
  typedef typename itk::ImageRegionIterator< InputImageType > RegionIteratorType; 
  RegionIteratorType RegionIterator( MaskImage, region );
  for ( RegionIterator.GoToBegin(); !RegionIterator.IsAtEnd(); ++RegionIterator)
  {
    RegionIterator.Set( 0.0 ); 
  }


  typename InputImageType::RegionType SphereRegion; 
  SphereRegion.SetSize( SizeOfSphereRegion ); 
  SphereRegion.SetIndex( BeginningOfSphereRegion ); 
  typedef itk::NeighborhoodIterator< InputImageType > NeighborhoodIteratorType;
  typename NeighborhoodIteratorType::RadiusType radius;
  radius.Fill( RadiusOfPatch );
  NeighborhoodIteratorType SphereRegionIterator( radius, MaskImage, SphereRegion ); 
  
  typename InputImageType::IndexType IndexWithinSphere; 
  for( unsigned int ii = 0; ii < IndicesWithinSphere.size(); ii++) 
  {
    SphereRegionIterator.SetPixel( IndicesWithinSphere[ ii ],  1.0 );
    //std::cout << "Writing index " << ii << " which is " << IndicesWithinSphere[ ii ] << std::endl;
  }

  return MaskImage;

}

/*PatchAnalysisObject::PatchAnalysisObject ( 
    double InputImage)
{

}
*/












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

template< unsigned int ImageDimension, class TRealType, class TImageType, 
  class TGradientImageType, class TInterpolator > 
vnl_vector< TRealType > ReorientPatchToReferenceFrame( 
    itk::NeighborhoodIterator< TImageType > GradientImageNeighborhood1, 
    itk::NeighborhoodIterator< TImageType > GradientImageNeighborhood2,
    const typename TImageType::Pointer MaskImage, 
    std::vector< unsigned int > IndicesWithinSphere, 
    std::vector< double > IndexWeights, 
    const typename TGradientImageType::Pointer GradientImage1, 
    const typename TGradientImageType::Pointer GradientImage2,
    unsigned int NumberOfValuesPerVoxel,
    TInterpolator Interpolator 
    )
{
  /* This function takes a reference patch and a moving patch and rotates 
   * the moving patch to match the reference patch.  
   * It returns an image equal to Image1, but with the reoriented entries from 
   * the moving patch inserted in place of the reference patch. 
   * Intended usage is to feed in an eigenvector in a canonical coordinate 
   * frame, generated by GenerateMaskImageFromPatch, that consists of only the 
   * entries in the eigenvector on a blank background.  The output of this function 
   * then is the moving neighborhood reoriented to match the input eigenvector. */

  typedef TRealType RealType; 
  typedef typename TImageType::PointType PointType; 
  typedef itk::CovariantVector< RealType, ImageDimension > GradientPixelType; 
  typedef vnl_vector< RealType > VectorType; 
  typedef typename TImageType::IndexType IndexType; 
  unsigned int NumberOfIndicesWithinSphere = IndicesWithinSphere.size();
  std::vector< PointType > ImagePatch1; 
  std::vector< PointType > ImagePatch2; 
  VectorType VectorizedImagePatch1( NumberOfIndicesWithinSphere, 0 ); 
  VectorType VectorizedImagePatch2( NumberOfIndicesWithinSphere, 0 ); 
  vnl_matrix< RealType > GradientMatrix1( NumberOfIndicesWithinSphere, NumberOfValuesPerVoxel ); 
  vnl_matrix< RealType > GradientMatrix2( NumberOfIndicesWithinSphere, NumberOfValuesPerVoxel ); 
  GradientMatrix1.fill( 0 ); 
  GradientMatrix2.fill( 0 ); 
  
  /*  Calculate center of each image patch so that rotations are about the origin. */
  PointType CenterPointOfImage1; 
  PointType CenterPointOfImage2; 
  CenterPointOfImage1.Fill( 0 ); 
  CenterPointOfImage2.Fill( 0 ); 
  RealType MeanNormalizingConstant = 1.0 / ( RealType ) NumberOfIndicesWithinSphere; 
  for( unsigned int ii = 0; ii < NumberOfIndicesWithinSphere; ii++ ) 
  {
    VectorizedImagePatch1[ ii ] = GradientImageNeighborhood1.GetPixel( IndicesWithinSphere[ ii ] );
    VectorizedImagePatch2[ ii ] = GradientImageNeighborhood2.GetPixel( IndicesWithinSphere[ ii ] ); 
    IndexType GradientImageIndex1 = GradientImageNeighborhood1.GetIndex( IndicesWithinSphere[ ii ] ); 
    IndexType GradientImageIndex2 = GradientImageNeighborhood2.GetIndex( IndicesWithinSphere[ ii ] ); 
    if( ( IsInside< TGradientImageType >( GradientImage1, GradientImageIndex1) ) && 
	( IsInside< TGradientImageType >( GradientImage2, GradientImageIndex2 ) ) )
    {
      GradientPixelType GradientPixel1 = GradientImage1->GetPixel( GradientImageIndex1 ) * IndexWeights[ ii ]; 
      GradientPixelType GradientPixel2 = GradientImage2->GetPixel( GradientImageIndex2 ) * IndexWeights[ ii ]; 
      for( unsigned int jj = 0; jj < NumberOfValuesPerVoxel; jj++)
      {
	GradientMatrix1( ii, jj ) = GradientPixel1[ jj ];
	GradientMatrix2( ii, jj ) = GradientPixel2[ jj ];
      }
      PointType Point1; 
      PointType Point2; 
      GradientImage1->TransformIndexToPhysicalPoint( GradientImageIndex1, Point1 ); 
      GradientImage2->TransformIndexToPhysicalPoint( GradientImageIndex2, Point2 ); 
      for( unsigned int dd = 0; dd < ImageDimension; dd++ )
      {
	CenterPointOfImage1[ dd ] = CenterPointOfImage1[ dd ] + Point1[ dd ] * MeanNormalizingConstant; 
	CenterPointOfImage2[ dd ] = CenterPointOfImage2[ dd ] + Point2[ dd ] * MeanNormalizingConstant; 
      }
      ImagePatch1.push_back( Point1 ); 
      ImagePatch2.push_back( Point2 ); 
    }
    else return vnl_vector< TRealType > (1, 0.0 ); 
  }
  RealType MeanOfImagePatch1 = VectorizedImagePatch1.mean(); 
  RealType MeanOfImagePatch2 = VectorizedImagePatch2.mean(); 
  VectorType CenteredVectorizedImagePatch1 = ( VectorizedImagePatch1 - MeanOfImagePatch1 ); 
  VectorType CenteredVectorizedImagePatch2 = ( VectorizedImagePatch2 - MeanOfImagePatch2 ); 
  RealType StDevOfImage1 = sqrt( CenteredVectorizedImagePatch1.squared_magnitude()  ); 
  RealType StDevOfImage2 = sqrt( CenteredVectorizedImagePatch2.squared_magnitude() ); 
  RealType correlation = inner_product( CenteredVectorizedImagePatch1, 
      CenteredVectorizedImagePatch2 ) / ( StDevOfImage1 * StDevOfImage2 );

  bool OK = true;
/*  std::cout << "VectorizedImagePatch1 is (before rotation) " << VectorizedImagePatch1 << std::endl;
  std::cout << "VectorizedImagePatch2 is (before rotation) " << VectorizedImagePatch2 << std::endl;*/
/*  std::cout << "GradientMatrix1 is " << GradientMatrix1 << std::endl; 
  std::cout << "GradientMatrix2 is " << GradientMatrix2 << std::endl; */
  vnl_matrix< RealType > CovarianceMatrixOfImage1 = GradientMatrix1.transpose() * GradientMatrix1; 
  vnl_matrix< RealType > CovarianceMatrixOfImage2 = GradientMatrix2.transpose() * GradientMatrix2; 
  vnl_symmetric_eigensystem< RealType > EigOfImage1( CovarianceMatrixOfImage1 ); 
  vnl_symmetric_eigensystem< RealType > EigOfImage2( CovarianceMatrixOfImage2 ); 
/*  std::cout << "CovarianceMatrixOfImage1 is " << CovarianceMatrixOfImage1 << std::endl; 
  std::cout << "CovarianceMatrixOfImage2 is " << CovarianceMatrixOfImage2 << std::endl;*/
  int NumberOfEigenvectors = EigOfImage1.D.cols(); 
  // FIXME: needs bug checking to make sure this is right
  // not sure how many eigenvectors there are or how they're indexed
  vnl_vector< RealType > Image1Eigvec1 = EigOfImage1.get_eigenvector( NumberOfEigenvectors - 1 ); // 0-indexed
  vnl_vector< RealType > Image1Eigvec2 = EigOfImage1.get_eigenvector( NumberOfEigenvectors - 2 ); 
  vnl_vector< RealType > Image2Eigvec1 = EigOfImage2.get_eigenvector( NumberOfEigenvectors - 1 ); 
  vnl_vector< RealType > Image2Eigvec2 = EigOfImage2.get_eigenvector( NumberOfEigenvectors - 2 ); 

  /* Solve Wahba's problem using Kabsch algorithm: 
   * arg_min(Q) \sum_k || w_k - Q v_k ||^2
   * Q is rotation matrix, w_k and v_k are vectors to be aligned.
   * Solution:  Denote B = \sum_k w_k v_k^T
   * Decompose B = U * S * V^T 
   * Then Q = U * M * V^T, where M = diag[ 1 1 det(U) det(V) ] 
   * Refs: http://journals.iucr.org/a/issues/1976/05/00/a12999/a12999.pdf 
   *       http://www.control.auc.dk/~tb/best/aug23-Bak-svdalg.pdf */
  vnl_matrix< RealType > B = outer_product( Image1Eigvec1, Image2Eigvec1 );
 /* std::cout << "Image1Eigvec1 is " <<  Image1Eigvec1 << std::endl;
  std::cout << "Image1Eigvec2 " << Image1Eigvec2 << std::endl;
  std::cout << "Image2Eigvec1 is " << Image2Eigvec1 << std::endl;
  std::cout << "Image2Eigvec2 is " << Image2Eigvec2 << std::endl;*/
  if( ImageDimension == 3)
  {
    B = outer_product( Image1Eigvec1, Image2Eigvec1 ) + 
        outer_product( Image1Eigvec2, Image2Eigvec2 ); 
  }
  vnl_svd< RealType > WahbaSVD( B );
  std::cout << "B is " << B << std::endl;
  vnl_matrix< RealType > Q_solution = WahbaSVD.V() * WahbaSVD.U().transpose();
  std::cout << "Q_solution is " << Q_solution << std::endl;
  // Now rotate the points to the same frame and sample neighborhoods again.
  for( unsigned int ii = 0; ii < NumberOfIndicesWithinSphere; ii++ )
  {
    PointType RotatedPoint = ImagePatch2[ ii ]; 
    // We also need vector representation of the point values
    vnl_vector< RealType > RotatedPointVector( RotatedPoint.Size(), 0 );
    // First move center of Patch 1 to center of Patch 2
    for( unsigned int dd = 0; dd < ImageDimension; dd++ )
    {
      RotatedPoint[ dd ] -= CenterPointOfImage2[ dd ]; 
      RotatedPointVector[ dd ] = RotatedPoint[ dd ]; 
    }
    // Now rotate RotatedPoint
    RotatedPointVector = ( Q_solution ) * RotatedPointVector; 
    for( unsigned int dd = 0; dd < ImageDimension; dd++ )
    {
      RotatedPoint[ dd ] = RotatedPointVector[ dd ] + CenterPointOfImage2[ dd ];
    } 
//    std::cout << "Original Point is " << ImagePatch2[ii] << ", Reoriented is " << RotatedPoint << std::endl;
    if( Interpolator->IsInsideBuffer( RotatedPoint) )
    {
      VectorizedImagePatch2[ ii ] = Interpolator->Evaluate( RotatedPoint );
    }
    else OK = false; 
  }
//  std::cout << "VectorizedImagePatch2 is " <<  VectorizedImagePatch2 << std::endl;
  // Generate image to return
  typename  TImageType::Pointer ReorientedImage = TImageType::New();
  ReorientedImage = ConvertVectorToSpatialImage<TImageType, 
		  TImageType, double > (VectorizedImagePatch2, MaskImage); 
  /*typedef itk::NeighborhoodIterator< TImageType > NeighborhoodIteratorType;
  NeighborhoodIteratorType ReorientedRegionIterator( 
      GradientImageNeighborhood1.GetRadius(), ReorientedImage, //GradientImage1, 
      GradientImageNeighborhood1.GetBoundingBoxAsImageRegion()); 

  for( unsigned int ii = 0; ii < IndicesWithinSphere.size(); ii++)
  {
    ReorientedRegionIterator.SetPixel( IndicesWithinSphere[ ii ],  
	VectorizedImagePatch2[ ii ] );
  }*/

//  return ReorientedImage; 
    return VectorizedImagePatch2; 
}





template< unsigned int ImageDimension, class TRealType, class TImageType, 
  class TGradientImageType, class TInterpolator > 
  double PatchCorrelation( 
      itk::NeighborhoodIterator< TImageType > GradientImageNeighborhood1, 
      itk::NeighborhoodIterator< TImageType > GradientImageNeighborhood2, 
      std::vector< unsigned int > IndicesWithinSphere, 
      std::vector< TRealType > IndexWeights, 
      typename TGradientImageType::Pointer GradientImage1, 
      typename TGradientImageType::Pointer GradientImage2, 
      TInterpolator Interpolator )
{
  typedef TRealType RealType; 
  typedef typename TImageType::PointType PointType; 
  typedef itk::CovariantVector< RealType, ImageDimension > GradientPixelType; 
  typedef vnl_vector< RealType > VectorType; 
  typedef typename TImageType::IndexType IndexType; 
  unsigned int NumberOfIndicesWithinSphere = IndicesWithinSphere.size();
  std::vector< PointType > ImagePatch1; 
  std::vector< PointType > ImagePatch2; 
  VectorType VectorizedImagePatch1( NumberOfIndicesWithinSphere, 0 ); 
  VectorType VectorizedImagePatch2( NumberOfIndicesWithinSphere, 0 ); 
  vnl_matrix< RealType > GradientMatrix1( NumberOfIndicesWithinSphere, ImageDimension ); 
  vnl_matrix< RealType > GradientMatrix2( NumberOfIndicesWithinSphere, ImageDimension ); 
  GradientMatrix1.fill( 0 ); 
  GradientMatrix2.fill( 0 ); 
  
  /*  Calculate center of each image patch so that rotations are about the origin. */
  PointType CenterPointOfImage1; 
  PointType CenterPointOfImage2; 
  CenterPointOfImage1.Fill( 0 ); 
  CenterPointOfImage2.Fill( 0 ); 
  RealType MeanNormalizingConstant = 1.0 / ( RealType ) NumberOfIndicesWithinSphere; 
  for( unsigned int ii = 0; ii < NumberOfIndicesWithinSphere; ii++ ) 
  {
    VectorizedImagePatch1[ ii ] = GradientImageNeighborhood1.GetPixel( IndicesWithinSphere[ ii ] );
    VectorizedImagePatch2[ ii ] = GradientImageNeighborhood2.GetPixel( IndicesWithinSphere[ ii ] ); 
    IndexType GradientImageIndex1 = GradientImageNeighborhood1.GetIndex( IndicesWithinSphere[ ii ] ); 
    IndexType GradientImageIndex2 = GradientImageNeighborhood2.GetIndex( IndicesWithinSphere[ ii ] ); 
    if( ( IsInside< TGradientImageType >( GradientImage1, GradientImageIndex1) ) && 
	( IsInside< TGradientImageType >( GradientImage2, GradientImageIndex2 ) ) )
    {
      GradientPixelType GradientPixel1 = GradientImage1->GetPixel( GradientImageIndex1 ) * IndexWeights[ ii ]; 
      GradientPixelType GradientPixel2 = GradientImage2->GetPixel( GradientImageIndex2 ) * IndexWeights[ ii ]; 
      for( unsigned int jj = 0; jj < ImageDimension; jj++)
      {
	GradientMatrix1( ii, jj ) = GradientPixel1[ jj ];
	GradientMatrix2( ii, jj ) = GradientPixel2[ jj ];
      }
      PointType Point1; 
      PointType Point2; 
      GradientImage1->TransformIndexToPhysicalPoint( GradientImageIndex1, Point1 ); 
      GradientImage2->TransformIndexToPhysicalPoint( GradientImageIndex2, Point2 ); 
      for( unsigned int dd = 0; dd < ImageDimension; dd++ )
      {
	CenterPointOfImage1[ dd ] = CenterPointOfImage1[ dd ] + Point1[ dd ] * MeanNormalizingConstant; 
	CenterPointOfImage2[ dd ] = CenterPointOfImage2[ dd ] + Point2[ dd ] * MeanNormalizingConstant; 
      }
      VectorizedImagePatch1.push_back( Point1 ); 
      VectorizedImagePatch2.push_back( Point2 ); 
    }
    else return 0; 
  }
  RealType MeanOfImagePatch1 = VectorizedImagePatch1.mean(); 
  RealType MeanOfImagePatch2 = VectorizedImagePatch2.mean(); 
  VectorType CenteredVectorizedImagePatch1 = ( VectorizedImagePatch1 - MeanOfImagePatch1 ); 
  VectorType CenteredVectorizedImagePatch2 = ( VectorizedImagePatch2 - MeanOfImagePatch2 ); 
  RealType StDevOfImage1 = sqrt( CenteredVectorizedImagePatch1 - MeanOfImagePatch1 ); 
  RealType StDevOfImage2 = sqrt( CenteredVectorizedImagePatch2 - MeanOfImagePatch2 ); 
  RealType correlation = inner_product( CenteredVectorizedImagePatch1, 
      CenteredVectorizedImagePatch2 ) / ( StDevOfImage1 * StDevOfImage2 );

  bool OK = true; 
  
  vnl_matrix< RealType > CovarianceMatrixOfImage1 = GradientMatrix1.transpose() * GradientMatrix1; 
  vnl_matrix< RealType > CovarianceMatrixOfImage2 = GradientMatrix2.transpose() * GradientMatrix2; 
  vnl_symmetric_eigensystem< RealType > EigOfImage1( CovarianceMatrixOfImage1 ); 
  vnl_symmetric_eigensystem< RealType > EigOfImage2( CovarianceMatrixOfImage2 ); 

  int NumberOfEigenvectors = EigOfImage1.D.cols(); 
  // FIXME: needs bug checking to make sure this is right
  // not sure how many eigenvectors there are or how they're indexed
  vnl_vector< RealType > Image1Eigvec1 = EigOfImage1.get_eigenvector( NumberOfEigenvectors - 1 ); // 0-indexed
  vnl_vector< RealType > Image1Eigvec2 = EigOfImage1.get_eigenvector( NumberOfEigenvectors - 2 ); 
  vnl_vector< RealType > Image2Eigvec1 = EigOfImage2.get_eigenvector( NumberOfEigenvectors - 1 ); 
  vnl_vector< RealType > Image2Eigvec2 = EigOfImage2.get_eigenvector( NumberOfEigenvectors - 2 ); 

  /* Solve Wahba's problem using Kabsch algorithm: 
   * arg_min(Q) \sum_k || w_k - Q v_k ||^2
   * Q is rotation matrix, w_k and v_k are vectors to be aligned.
   * Solution:  Denote B = \sum_k w_k v_k^T
   * Decompose B = U * S * V^T 
   * Then Q = U * M * V^T, where M = diag[ 1 1 det(U) det(V) ] 
   * Refs: http://journals.iucr.org/a/issues/1976/05/00/a12999/a12999.pdf 
   *       http://www.control.auc.dk/~tb/best/aug23-Bak-svdalg.pdf */
  vnl_matrix< RealType > B = outer_product( Image1Eigvec1, Image2Eigvec1 ); 
  if( ImageDimension == 3)
  {
    B = outer_product( Image1Eigvec1, Image2Eigvec1 ) + 
        outer_product( Image1Eigvec2, Image2Eigvec2 ); 
  }
  vnl_svd< RealType > WahbaSVD( B ); 
  vnl_matrix< RealType > Q_solution = WahbaSVD.V() * WahbaSVD.U().transpose(); 
  // Now rotate the points to the same frame and sample neighborhoods again.
  for( unsigned int ii = 0; ii < NumberOfIndicesWithinSphere; ii++ )
  {
    PointType RotatedPoint = ImagePatch2[ ii ]; 
    // We also need vector representation of the point values
    vnl_vector< RealType > RotatedPointVector( RotatedPoint.Size(), 0 );
    // First move center of Patch 1 to center of Patch 2
    for( unsigned int dd = 0; dd < ImageDimension; dd++ )
    {
      RotatedPoint[ dd ] -= CenterPointOfImage2[ dd ]; 
      RotatedPointVector[ dd ] = RotatedPoint[ dd ]; 
    }
    // Now rotate RotatedPoint
    RotatedPointVector = ( Q_solution ) * RotatedPointVector; 
    for( unsigned int dd = 0; dd < ImageDimension; dd++ )
    {
      RotatedPoint[ dd ] = RotatedPointVector + CenterPointOfImage2[ dd ];
    }
    if( Interpolator->IsInsideBuffer( RotatedPoint) )
    {
      VectorizedImagePatch2[ ii ] = Interpolator->Evaluate( RotatedPoint );
    }
    else OK = false; 
  }
  if( OK )
  {
    MeanOfImagePatch2 = VectorizedImagePatch2.mean(); 
    CenteredVectorizedImagePatch2 = ( VectorizedImagePatch2 - MeanOfImagePatch2 ); 
    StDevOfImage2 = sqrt( CenteredVectorizedImagePatch2.squared_magnitude() ); 
    correlation = inner_product( VectorizedImagePatch1, VectorizedImagePatch2 ) / 
      (StDevOfImage1 * StDevOfImage2 ); 
  }
  else correlation = 0; 

  if ( vnl_math_isnan( correlation ) || vnl_math_isinf( correlation )  ) return 0; 
  else return correlation;
}


