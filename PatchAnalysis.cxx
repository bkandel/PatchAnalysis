#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "vnl/vnl_matrix.h"
#include "itkMatrix.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkCSVNumericObjectFileWriter.h"
#include <vnl/algo/vnl_svd.h>

using namespace std; 

int main(int, char * argv[] )
{
  typedef float       InputPixelType; 
  const unsigned int  Dimension = 3; // assume 3d images 
  const unsigned int  NumberOfPatches = 1000; 
  const unsigned int  SizeOfPatches = 5;
  const unsigned int  VolumeOfPatches = 125; // illegal: pow(SizeOfPatches, Dimension);  

  typedef itk::Image< InputPixelType, Dimension >   InputImageType;
  InputImageType::Pointer InputImage = InputImageType::New();
  InputImageType::Pointer MaskImage  = InputImageType::New();  
  InputImageType::Pointer PatchImage = InputImageType::New(); 

  typedef itk::Image< InputPixelType, 2 > VectorizedPatchImageType; 
  VectorizedPatchImageType::Pointer VectorizedPatchImage = 
                                    VectorizedPatchImageType::New(); 

  typedef itk::ImageFileReader< InputImageType > ReaderType;
  ReaderType::Pointer  inputImageReader = ReaderType::New();
  ReaderType::Pointer  maskImageReader  = ReaderType::New(); 
  typedef itk::CSVNumericObjectFileWriter< InputPixelType, NumberOfPatches, VolumeOfPatches > WriterType; 
  WriterType::Pointer patchWriter = WriterType::New(); 
  WriterType::Pointer eigvecWriter = WriterType::New(); 

/*  typedef itk::Matrix< InputPixelType, 
                       NumberOfPatches, 
                       VolumeOfPatches > MatrixType; 
  MatrixType VectorizedPatchMatrix;*/
  vnl_matrix< InputPixelType > VectorizedPatchMatrix(NumberOfPatches, VolumeOfPatches); 
  VectorizedPatchMatrix.fill( 0 );  

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
    TestPatchSeed( 0 ) = rand() % inputSize[0]; 
    TestPatchSeed( 1 ) = rand() % inputSize[1]; 
    TestPatchSeed( 2 ) = rand() % inputSize[2];
    PatchIndex[0] = TestPatchSeed(0);
    PatchIndex[1] = TestPatchSeed(1); 
    PatchIndex[2] = TestPatchSeed(2); 
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
  
  for( int i = 0; i < NumberOfPatches; ++i)
  {
    int j = 0; 
    PatchIndex[ 0 ] = PatchSeedPoints( i, 0 );
    PatchIndex[ 1 ] = PatchSeedPoints( i, 1 );  
    PatchIndex[ 2 ] = PatchSeedPoints( i, 2 );
    InputImageType::RegionType desiredPatch( PatchIndex, PatchSize ); 
    extractFilter->SetRegionOfInterest( desiredPatch );
    extractFilter->SetInput( inputImageReader->GetOutput() );
    extractFilter->Update();  
    PatchImage = extractFilter->GetOutput(); 
    typedef itk::ImageRegionIteratorWithIndex< InputImageType > Iterator; 
    Iterator patchIterator( PatchImage, 
                            PatchImage->GetLargestPossibleRegion() ); 
    for( patchIterator.GoToBegin(); !patchIterator.IsAtEnd(); ++patchIterator)
    {
      InputPixelType PatchValue = patchIterator.Get(); 
      VectorizedPatchMatrix( i, j ) = PatchValue; 
      ++j;  
    }
  }
  vnl_svd< InputPixelType > svd( VectorizedPatchMatrix ); 
  vnl_matrix< InputPixelType > PatchEigenvectors = svd.V();  
  patchWriter->SetFileName( outputFilename ); 
  patchWriter->SetInput( &VectorizedPatchMatrix ); 
  patchWriter->Update(); 

  eigvecWriter->SetFileName( eigvecFilename ); 
  eigvecWriter->SetInput( &PatchEigenvectors); 
  eigvecWriter->Update(); 

  return 0;   
}
