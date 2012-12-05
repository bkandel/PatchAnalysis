#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <time.h>
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

using namespace std; 

int main(int argc, char * argv[] )
{
  if( argc < 5) 
  {
    cout << "Usage: " << argv[0] << 
       " InputFilename MaskFilename VectorizedPatchFilename EigenvectorFilename " << endl; 
    return 1; 
  }
  typedef double       InputPixelType; 
  const unsigned int  Dimension = 2; // assume 2d images 
  const unsigned int  NumberOfPatches = 1000; 
  const unsigned int  SizeOfPatches = 3;
  const unsigned int  VolumeOfPatches = 49; //343; // illegal: pow(SizeOfPatches, Dimension);  
  double TargetPercentVarianceExplained = 0.95; 

  typedef itk::Image< InputPixelType, Dimension >   InputImageType;
  InputImageType::Pointer InputImage = InputImageType::New();
  InputImageType::Pointer MaskImage  = InputImageType::New();  
  InputImageType::Pointer PatchImage = InputImageType::New(); 

  typedef itk::ImageFileReader< InputImageType > ReaderType;
  ReaderType::Pointer  inputImageReader = ReaderType::New();
  ReaderType::Pointer  maskImageReader  = ReaderType::New(); 
  typedef itk::CSVNumericObjectFileWriter< InputPixelType, 
                         NumberOfPatches, VolumeOfPatches > WriterType; 
  WriterType::Pointer patchWriter = WriterType::New(); 
  WriterType::Pointer eigvecWriter = WriterType::New(); 


  const char * inputFilename = argv[1];
  const char * maskFilename  = argv[2]; 
  const char * outputFilename = argv[3];
  const char * eigvecFilename = argv[4]; 
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
  typedef typename InputImageType::IndexType IndexType;
  IndexType PatchCenterIndex;
  PatchCenterIndex.Fill( SizeOfPatches );
  Iterator.SetLocation( PatchCenterIndex );
  // get indices within N-d sphere
  vector< unsigned int > IndicesWithinSphere;
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
      VectorizedPatchMatrix( i, j ) = Iterator.GetPixel( j ); 
    }
  }
  
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
  
  patchWriter->SetFileName( outputFilename ); 
  patchWriter->SetInput( &VectorizedPatchMatrix ); 
  patchWriter->Update(); 
  eigvecWriter->SetFileName( eigvecFilename ); 
  eigvecWriter->SetInput( &PatchEigenvectors); 
  eigvecWriter->Update(); 

  eigvecWriter->SetFileName( SignificantEigvecFilename ); 
  eigvecWriter->SetInput( &SignificantPatchEigenvectors ); 
  eigvecWriter->Update();
  
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
      PatchesForAllPointsWithinMask( j, i ) = Iterator.GetPixel( j );
    }
  }
  cout << "Recorded patches for all points." << endl;

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
//    cout << "PatchOfInterest: " << PatchOfInterest << endl; 
    vnl_vector< InputPixelType > x( NumberOfSignificantEigenvectors ); 
    x.fill( 0 );
//    if(i % 100000 == 0) cout << "Computed " << i << " out of " << SumOfMaskImage << "regressions." << endl;
    
    x = RegressionSVD.solve(PatchOfInterest); 
    EigenvectorCoefficients.set_column(i, x);
  }
  patchWriter->SetFileName( "eigenvectorCoefficients.csv" );
  patchWriter->SetInput( &EigenvectorCoefficients );
  patchWriter->Update();
  vnl_matrix< InputPixelType > ReconstructedPatches = SignificantPatchEigenvectors * EigenvectorCoefficients; 
  vnl_matrix< InputPixelType > Error = ReconstructedPatches - PatchesForAllPointsWithinMask; 
  cout << "Error is " << Error.array_two_norm() << " as compared to norm of " 
    << PatchesForAllPointsWithinMask.absolute_value_sum() << "." << endl;
  cout << "Done!" << endl;
  return 0;   
}
