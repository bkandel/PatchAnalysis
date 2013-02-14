#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "vnl/vnl_matrix.h"
#include "itkNeighborhoodIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkCSVNumericObjectFileWriter.h"
#include <vnl/algo/vnl_svd.h>
#include <itkStatisticsImageFilter.h>
#include "itkImageRegionIterator.h"
#include "PatchAnalysis.hxx"
#include "PatchAnalysis.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkGradientImageFilter.h"
#include "itkCovariantVector.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
using namespace std; 

int main(int argc, char * argv[] )
{
  if( argc < 7) 
  {
    cout << "Usage: " << argv[0] << 
       " InputFilename MaskFilename VectorizedPatchFilename EigenvectorFilename SizeOfPatches TargetVarianceExplained" << endl; 
    return 1; 
  }
  typedef double      InputPixelType; 
  typedef double      RealType; 
  const unsigned int  Dimension = 2; // assume 2d images 
  const unsigned int  NumberOfPatches = 1000; 

  typedef itk::Image< InputPixelType, Dimension >   InputImageType;
  InputImageType::Pointer InputImage;
  InputImageType::Pointer MaskImage;  
  InputImageType::Pointer PatchImage; 

  typedef itk::ImageFileReader< InputImageType > ReaderType;
  ReaderType::Pointer  inputImageReader = ReaderType::New();
  ReaderType::Pointer  maskImageReader  = ReaderType::New(); 

  const char * inputFilename            = argv[1];
  const char * maskFilename             = argv[2]; 
  const char * outputFilename           = argv[3];
  const char * eigvecFilename           = argv[4]; 
  const unsigned int  SizeOfPatches     = atoi(argv[ 5 ]);
  const unsigned int  VolumeOfPatches   = pow(SizeOfPatches, Dimension); //49; //343; // illegal: pow(SizeOfPatches, Dimension);  
  double TargetPercentVarianceExplained = atof( argv[ 6 ] ); 
  inputImageReader->SetFileName( inputFilename );
  inputImageReader->Update();
  maskImageReader->SetFileName( maskFilename ); 
  maskImageReader->Update(); 
  InputImage = inputImageReader->GetOutput(); 
  MaskImage  = maskImageReader->GetOutput(); 

  InputImageType::SizeType inputSize =
       InputImage->GetLargestPossibleRegion().GetSize();

  typedef itk::RegionOfInterestImageFilter< InputImageType,
                                            InputImageType > ExtractFilterType;
 
  ExtractFilterType::Pointer extractFilter = ExtractFilterType::New();
  srand( time( NULL) ); 
  // define region sizes
  InputImageType::SizeType PatchSize; 
  PatchSize.Fill( SizeOfPatches ); 
  //randomly sample points inside image
  vnl_matrix< int > PatchSeedPoints( NumberOfPatches, Dimension );
  vnl_vector< int > TestPatchSeed( Dimension );  
  int  PatchSeedIterator = 0;
  int  PatchSeedAttemptIterator = 0; 
  InputImageType::IndexType PatchIndex;  
  while( PatchSeedIterator < NumberOfPatches) 
  {
    for( int i = 0; i < Dimension; ++i)
    {
      PatchIndex[ i ] = TestPatchSeed( i ) = rand() % inputSize[ i ]; 
    }
    InputPixelType MaskValue = MaskImage->GetPixel( PatchIndex ); 
    if( MaskValue > 0 )
    {
      PatchSeedPoints.set_row( PatchSeedIterator, TestPatchSeed ); 
      ++PatchSeedIterator;  
    } 
    ++PatchSeedAttemptIterator; 
  }
  
  cout << "Found " << PatchSeedIterator << 
          " points in " << PatchSeedAttemptIterator << 
          " attempts." << endl;

  
  // allocate matrix based on radial size of patch
  int i = 0; 
  for( int j = 0; j < Dimension; ++j)
  {
    PatchIndex[ j ] = PatchSeedPoints( i, j ); 
  } 
  typedef itk::NeighborhoodIterator< InputImageType > IteratorType;
  IteratorType::RadiusType radius;
  radius.Fill( SizeOfPatches );
  IteratorType Iterator( radius, inputImageReader->GetOutput(),
                           inputImageReader->GetOutput()->GetRequestedRegion() );
  typedef InputImageType::IndexType IndexType;
  IndexType PatchCenterIndex;
  PatchCenterIndex.Fill( SizeOfPatches );
  Iterator.SetLocation( PatchCenterIndex );

  // get indices within N-d sphere
  vector< unsigned int > IndicesWithinSphere;
  vector< double > Weights; 
  for( int ii = 0; ii < Iterator.Size(); ++ii)
  {
    IndexType Index = Iterator.GetIndex( ii );
    float DistanceFromPatchCenter = 0.0;
    for( int jj = 0; jj < Dimension; ++jj)
    {
      DistanceFromPatchCenter +=
         ( Index[jj] - PatchCenterIndex[jj] ) *
         ( Index[jj] - PatchCenterIndex[jj] );
    }
    DistanceFromPatchCenter = sqrt(DistanceFromPatchCenter);
    if( DistanceFromPatchCenter <= SizeOfPatches )
    {
      IndicesWithinSphere.push_back( ii );
      Weights.push_back( 1.0 ); 
    }
  }
  cout << Iterator.Size() << endl;
  cout << IndicesWithinSphere.size() << endl;

  // populate matrix with patch values from points in image
  vnl_matrix< InputPixelType > VectorizedPatchMatrix( NumberOfPatches, IndicesWithinSphere.size() ); 
  VectorizedPatchMatrix.fill( 0 );  
  for( int i = 0; i < NumberOfPatches; ++i)
  {
    for( int j = 0; j < Dimension; ++j)
    {
      PatchCenterIndex[ j ] = PatchSeedPoints( i, j ); 
    }
    Iterator.SetLocation( PatchCenterIndex ); 
    // get indices within N-d sphere
    for( int j = 0; j < IndicesWithinSphere.size(); ++j)
    {
      VectorizedPatchMatrix( i, j ) = Iterator.GetPixel( IndicesWithinSphere[ j ] );
    }
  }
  cout << "VectorizedPatchMatrix is " << VectorizedPatchMatrix.rows() << 
    "x" << VectorizedPatchMatrix.cols() << "." << endl;
  //compute eigendecomposition of patch matrix
  vnl_svd< InputPixelType > svd( VectorizedPatchMatrix ); 
  vnl_matrix< InputPixelType > PatchEigenvectors = svd.V();  
  cout << "PatchEigenvectors is " << PatchEigenvectors.rows() << 
    "x" << PatchEigenvectors.columns() << "." << endl;
  double SumOfEigenvalues = 0.0; 
  for( int i = 0; i < svd.rank(); i++)
  {
    SumOfEigenvalues += svd.W(i, i); 
  }
  double PartialSumOfEigenvalues = 0.0; 
  double PercentVarianceExplained = 0.0; 
  i = 0; 
  while( PercentVarianceExplained < TargetPercentVarianceExplained && i < svd.rank())
  {
    PartialSumOfEigenvalues += svd.W(i, i); 
    PercentVarianceExplained = PartialSumOfEigenvalues / 
                                      SumOfEigenvalues; 
    i++; 
  }
  int NumberOfSignificantEigenvectors = i; 
  cout << "It took " << NumberOfSignificantEigenvectors << " eigenvalues to reach " << 
       TargetPercentVarianceExplained * 100 << "% variance explained." << endl;
  vnl_matrix< InputPixelType > SignificantPatchEigenvectors; 
  SignificantPatchEigenvectors = PatchEigenvectors.get_n_columns(0, i);
  string SignificantEigvecFilename = "significantEigenvectors.csv";
  cout << "SignificantPatchEigenvectors is " << SignificantPatchEigenvectors.rows() << 
    "x" << SignificantPatchEigenvectors.columns() << "." << endl;
  
  // find total number of non-zero points in mask and store indices
  // WARNING: ASSUMES MASK IS BINARY!!
  typedef itk::StatisticsImageFilter< InputImageType > StatisticsFilterType; 
  StatisticsFilterType::Pointer StatisticsFilter = StatisticsFilterType::New(); 
  StatisticsFilter->SetInput(MaskImage); 
  StatisticsFilter->Update(); 
  double SumOfMaskImage = StatisticsFilter->GetSum();
  cout << "Total number of possible points: " << SumOfMaskImage << "." << endl;
  vnl_matrix< int > NonZeroMaskIndices( SumOfMaskImage, Dimension ); 
  typedef  itk::ImageRegionIterator< InputImageType > ImageIteratorType; 
  ImageIteratorType MaskImageIterator( MaskImage , MaskImage->GetLargestPossibleRegion());
  
  int MaskImagePointIter = 0;
  for(MaskImageIterator.GoToBegin(); !MaskImageIterator.IsAtEnd(); ++MaskImageIterator)
  {
    if( MaskImageIterator.Get() > 0 )
    {
      for( int i = 0; i < Dimension; i++)
      {
        NonZeroMaskIndices(MaskImagePointIter, i) = MaskImageIterator.GetIndex()[i]; 
      }
      MaskImagePointIter++; 
    }
  }
  cout << "Number of points is " << MaskImagePointIter << endl;

  
  // generate patches for all points in mask
  vnl_matrix< InputPixelType > PatchesForAllPointsWithinMask( IndicesWithinSphere.size(),  SumOfMaskImage);
  PatchesForAllPointsWithinMask.fill( 0 ); 
  for( int i = 0; i < SumOfMaskImage; ++i)
  {  
    for( int j = 0; j < Dimension; ++j)
    {
      PatchCenterIndex[ j ] = NonZeroMaskIndices(i, j);
    }
    Iterator.SetLocation( PatchCenterIndex );
    // get indices within N-d sphere
    for( int j = 0; j < IndicesWithinSphere.size(); ++j)
    {
      PatchesForAllPointsWithinMask( j, i ) = Iterator.GetPixel( IndicesWithinSphere[ j ] );
    }
  }
  cout << "Recorded patches for all points." << endl;

  // Compute correlation 
  


//  RotatedImage

  typedef itk::LinearInterpolateImageFunction<InputImageType,float> ScalarInterpolatorType;
  typedef typename ScalarInterpolatorType::Pointer InterpPointer;
  InterpPointer interp1 =  ScalarInterpolatorType::New();


  typedef itk::ImageFileWriter< InputImageType > ImageWriterType;

  InputImageType::Pointer EigvecMaskImage;
  EigvecMaskImage = GenerateMaskImageFromPatch< InputImageType, InputPixelType >( 
      IndicesWithinSphere, SizeOfPatches, Dimension);
  ImageWriterType::Pointer EigvecMaskImageWriter = ImageWriterType::New(); 
  EigvecMaskImageWriter->SetInput( EigvecMaskImage ); 
  EigvecMaskImageWriter->SetFileName( "TestMask.nii.gz" ); 
  EigvecMaskImageWriter->Update(); 
  ImageWriterType::Pointer EigvecWriter = ImageWriterType::New(); 
  // write out eigenvectors 
  for ( unsigned int ii = 0; ii < NumberOfSignificantEigenvectors; ii++)
  {
    vnl_vector< InputPixelType > EigvecAsPixel =
	        SignificantPatchEigenvectors.get_column( ii );
    string ImageIndex;
    ostringstream convert;
    convert << ii;
    ImageIndex = convert.str();
    EigvecWriter->SetInput( ConvertVectorToSpatialImage< InputImageType, 
	InputImageType, double >( EigvecAsPixel, 
	  EigvecMaskImage) );
    string EigvecFileName = "Eigvec" + ImageIndex + ".nii.gz" ; 
    EigvecWriter->SetFileName(EigvecFileName);  
    EigvecWriter->Update(); 
  }

  
  
  
  
  
  // output sample patch 
  int SamplePatchNumber = 400; 
  InputImageType::Pointer SamplePatchImage = InputImageType::New(); 
  vnl_vector< InputPixelType > SamplePatchVector =
	      PatchesForAllPointsWithinMask.get_column( SamplePatchNumber );  

  SamplePatchImage = ConvertVectorToSpatialImage< InputImageType, 
		   InputImageType, double >( SamplePatchVector,  
		       EigvecMaskImage ); 
  ImageWriterType::Pointer SamplePatchImageWriter = ImageWriterType::New(); 
  SamplePatchImageWriter->SetInput( SamplePatchImage ); 
  SamplePatchImageWriter->SetFileName( "SamplePatch.nii.gz" ); 
  SamplePatchImageWriter->Update(); 
  
  
  
  
  
  
  
  // test rotation of eigenvectors 
  int FixedIndex = 3; 
  int MovingIndex = 5;
  int NumberOfPaddingVoxels = 2; 
  int RadiusOfPatch = SizeOfPatches; 
  typedef itk::NeighborhoodIterator< InputImageType > NeighborhoodIteratorType;
  radius.Fill( RadiusOfPatch );
  typename InputImageType::RegionType SphereRegion;
  typename InputImageType::IndexType   BeginningOfSphereRegion;
  typename InputImageType::SizeType    SizeOfSphereRegion;
  typedef itk::CovariantVector<RealType, Dimension>                               GradientPixelType;
  typedef itk::Image<GradientPixelType, Dimension>                                GradientImageType;
  typedef itk::SmartPointer<GradientImageType>                                    GradientImagePointer;
  typedef itk::GradientRecursiveGaussianImageFilter<InputImageType, GradientImageType> GradientImageFilterType;
  typedef typename GradientImageFilterType::Pointer                               GradientImageFilterPointer;
  typedef itk::NeighborhoodIterator< GradientImageType >                          GradientNeighborhoodIteratorType; 

  RealType     GradientSigma = 1.0;
  GradientImageFilterPointer FixedGradientFilter = GradientImageFilterType::New(); 
  GradientImageFilterPointer MovingGradientFilter = GradientImageFilterType::New(); 

  for( unsigned int dd = 0; dd < Dimension; dd++)
  {
    BeginningOfSphereRegion[ dd ] = NumberOfPaddingVoxels + RadiusOfPatch; 
    SizeOfSphereRegion[ dd ] = RadiusOfPatch * 2 + 1;
  }

  SphereRegion.SetSize( SizeOfSphereRegion );
  SphereRegion.SetIndex( BeginningOfSphereRegion );
  
  NeighborhoodIteratorType RegionIterator( radius, EigvecMaskImage, SphereRegion );
  InputImageType::Pointer FixedImage; 
  InputImageType::Pointer MovingImage; 
  InputImageType::Pointer RotatedImage;
  ImageWriterType::Pointer RotationWriter = ImageWriterType::New(); 
  ImageWriterType::Pointer FixedWriter = ImageWriterType::New(); 
  ImageWriterType::Pointer MovingWriter = ImageWriterType::New(); 
  
  vnl_vector< InputPixelType > FixedVector = SignificantPatchEigenvectors.get_column(FixedIndex); 
  vnl_vector< InputPixelType > MovingVector = SignificantPatchEigenvectors.get_column(MovingIndex); 

  FixedImage = ConvertVectorToSpatialImage< InputImageType, 
             InputImageType, double > ( FixedVector,
                 EigvecMaskImage); 
  MovingImage = ConvertVectorToSpatialImage< InputImageType, 
              InputImageType, double > ( MovingVector, 
                  EigvecMaskImage);
  
  FixedGradientFilter->SetInput( FixedImage ); 
  FixedGradientFilter->SetSigma( GradientSigma ); 
  FixedGradientFilter->Update();
  GradientImageType::Pointer FixedGradientImage = FixedGradientFilter->GetOutput();

  MovingGradientFilter->SetInput( MovingImage ); 
  MovingGradientFilter->Update( ); 
  GradientImageType::Pointer MovingGradientImage = MovingGradientFilter->GetOutput(); 
  

  NeighborhoodIteratorType FixedIterator( radius, FixedImage, SphereRegion ); 
  NeighborhoodIteratorType MovingIterator( radius, MovingImage, SphereRegion ); 
  FixedWriter->SetInput(FixedImage); 
  FixedWriter->SetFileName("Fixed.nii.gz"); 
  FixedWriter->Update(); 
  MovingWriter->SetInput(MovingImage); 
  MovingWriter->SetFileName("Moving.nii.gz"); 
  MovingWriter->Update(); 
  InputImageType::Pointer ReorientedEigvec;
  unsigned int NumberOfValuesPerVoxel = 1;
  cout << "Weights are " << Weights.size() << endl;
  interp1->SetInputImage( MovingImage ); 
  ReorientedEigvec = 
    ReorientPatchToReferenceFrame< Dimension, InputPixelType, InputImageType, 
    GradientImageType, InterpPointer > (
	FixedIterator, 
	MovingIterator, 
        EigvecMaskImage,
	IndicesWithinSphere, 
	Weights,
	FixedGradientImage, 
	MovingGradientImage, 
	Dimension, 
	interp1
	);

  RotationWriter->SetInput(ReorientedEigvec); 
  RotationWriter->SetFileName("Rotated.nii.gz"); 
  RotationWriter->Update(); 



  // perform regression from eigenvectors to images
  // Ax = b, where A is eigenvector matrix (number of indices
  // within patch x number of eigenvectors), x is coefficients 
  // (number of eigenvectors x 1), b is patch values for a given index
  // (number of indices within patch x 1).
  cout << "Computing regression." << endl;
  vnl_matrix< InputPixelType > 
    EigenvectorCoefficients( NumberOfSignificantEigenvectors, SumOfMaskImage ); 
  EigenvectorCoefficients.fill( 0 );
  vnl_svd< InputPixelType > RegressionSVD(SignificantPatchEigenvectors);  
//  EigenvectorCoefficients =  RegressionSVD.solve(PatchesForAllPointsWithinMask); 
//  not feasible for large matrices
  for( int i = 0; i < SumOfMaskImage; ++i)
  {
    vnl_vector< InputPixelType > PatchOfInterest = 
      PatchesForAllPointsWithinMask.get_column(i);
    vnl_vector< InputPixelType > x( NumberOfSignificantEigenvectors ); 
    x.fill( 0 );
    x = RegressionSVD.solve(PatchOfInterest); 
    EigenvectorCoefficients.set_column(i, x);
  }
  vnl_matrix< InputPixelType > ReconstructedPatches = SignificantPatchEigenvectors * EigenvectorCoefficients; 
  vnl_matrix< InputPixelType > Error = ReconstructedPatches - PatchesForAllPointsWithinMask;
  vnl_vector< InputPixelType > PercentError(Error.columns() ); 
  for( int i = 0; i < Error.columns(); ++i)
  {
    PercentError(i) = Error.get_column(i).two_norm() / (PatchesForAllPointsWithinMask.get_column(i).two_norm() + 1e-10); 
  }
  cout << "Average percent error is " << PercentError.mean() * 100 << "%, with max of " << 
    PercentError.max_value() * 100 << "%." <<  endl;
  cout << "EigenvectorCoefficients is " << EigenvectorCoefficients.rows() << "x" << 
    EigenvectorCoefficients.columns() << "." << endl;
  
  InputImageType::Pointer ConvertedImage;
  for( int i = 0; i < EigenvectorCoefficients.rows(); ++i) 
  {
    vnl_vector< InputPixelType > RegressionCoefficient = 
      EigenvectorCoefficients.get_row(i); 
    ConvertedImage = ConvertVectorToSpatialImage<InputImageType,InputImageType,double>( 
      RegressionCoefficient, MaskImage);
    typedef itk::ImageFileWriter< InputImageType > ImageWriterType;
    ImageWriterType::Pointer  CoefficientImageWriter = ImageWriterType::New();
    CoefficientImageWriter->SetInput( ConvertedImage ); 
    string ImageIndex; 
    ostringstream convert; 
    convert << i; 
    ImageIndex = convert.str(); 
    string CoefficientImageFileName = "coeffs_out_" + ImageIndex + ".nii.gz"; 
    CoefficientImageWriter->SetFileName( CoefficientImageFileName ); 
    CoefficientImageWriter->Update(); 
  }

  
  return 0;   
}
