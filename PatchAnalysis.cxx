#include <iostream>
#include <fstream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sstream>
#include <unistd.h>
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
#include "itkLinearInterpolateImageFunction.h"
#include "itkGradientImageFilter.h"
#include "itkCovariantVector.h"
#include "itkGradientRecursiveGaussianImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "PatchAnalysis.h"
#include "PatchAnalysis.hxx"



using namespace std; 

void printHelp( void )
{
	cout << "COMMAND: " << endl;
	cout << "    PatchAnalysis" << endl;
	cout << "Learn dictionary of eigenpatches of an input image and output " << endl;
	cout << "projection of every patch in image onto eigenpatches." << endl;
	cout << "Inputs: " <<
			"Image to use for learning eigenpatches and projecting onto eigenpatches." << endl;
	cout << "Outputs:  Learned patches and projection images for all eigenpatches." << endl;
	cout << endl;
	cout << "OPTIONS:" << endl;
	cout << "    -i <inputName> " << endl;
	cout << "    -m <mask> " << endl;
	cout << "    -p [prefix for patch projection outputs]" << endl;
	cout << "    -e [prefix for eigenvector image outputs]  If not set, eigenvectors are not written." << endl;
	cout << "    -s [size of patches=3]" << endl;
	cout << "    -t [target variance explained=0.95]" <<
			" If greater than 1, number of eigenvectors to retain." << endl;
	cout << "    -n [number of sample patches to take=1000]" << endl;
	cout << "    -o compute orientation invariant eigenvectors and projections" << endl;
	cout << "    -f [name for output eigenvector matrix]" << endl;
	cout << "    -g [name for input eigenvector matrix]"  << endl;
	cout << "    -q [name for output patch matrix]"       << endl;
	cout << "    -v print verbose output" << endl;
	cout << "    -c mean center patches 0/1 (default=true)" << endl;
	exit( EXIT_FAILURE );
}

int main(int argc, char * argv[] )
{
  ArgumentType args;
  args.inputName               = "";
  args.maskName                = "";
  args.outProjectionName       = "projectedPatches";
  args.eigvecName              = "";
  args.patchSize               = 3;
  args.targetVarianceExplained = 0.95;
  args.outEigvecMatrixName     = "";
  args.inEigvecMatrixName      = "";
  args.numberOfSamplePatches   = 1000;
  args.verbose                 = 0;
  args.orientationInvariant    = false;
  args.outPatchName            = "";
  args.meanCenter              = true;
  args.help                    = 0;

  const char * optString = "i:m:p:e:s:t:f:g:n:q:c:voh";
  int opt = 0;
  while( (opt = getopt( argc, argv, optString)) != -1 )
  {
    switch( opt )
    {
    case 'i':
    	args.inputName = optarg;
    	break;
    case 'm':
    	args.maskName = optarg;
    	break;
    case 'p':
    	args.outProjectionName = optarg;
    	break;
    case 'e':
    	args.eigvecName = optarg;
    	break;
    case 's':
    	args.patchSize = atoi( optarg );
    	break;
    case 't':
    	args.targetVarianceExplained = atof( optarg );
    	break;
    case 'n':
    	args.numberOfSamplePatches = atoi( optarg );
    	break;
    case 'v':
    	args.verbose = 1;
    	break;
    case 'o':
    	args.orientationInvariant = true;
    	break;
    case 'f':
    	args.outEigvecMatrixName = optarg;
    	break;
    case 'g':
    	args.inEigvecMatrixName = optarg;
    	break;
    case 'q':
    	args.outPatchName = optarg;
    	break;
    case 'c':
    	if(atoi(optarg) == 0) {
    		args.meanCenter = false;
    	} else if(atoi(optarg) == 1){
    		args.meanCenter = true;
    	} else {
    		std::cout << "Error: c option must be 0 or 1" << std::endl;
    		printHelp();
    	}
    	break;
    case 'h':
    	printHelp();
    	break;
    default:
    	cout << "Error: Unrecognized command." << endl;
    	printHelp();
    	break;
    }
  }

  if( args.inputName.empty() || args.maskName.empty() )
  {
	  cout << "Error: Input image and mask name required." << endl;
	  cout << "For help, run " << argv[0] << " -h." << endl;
	  exit( EXIT_FAILURE ) ;
  }
  char inputNameChar[ args.inputName.size() + 1];
  char maskNameChar[ args.maskName.size() + 1 ];
  strcpy( inputNameChar, args.inputName.c_str() );
  strcpy( maskNameChar, args.maskName.c_str() );
  ifstream inputFile( inputNameChar );
  if( !inputFile.good() )
  {
	  cout << "Error: Input image does not exist." << endl;
	  exit( EXIT_FAILURE );
  }
  inputFile.close();
  ifstream maskFile( maskNameChar );
  if (!maskFile.good() )
  {
	  cout << "Error: Mask image does not exist." << endl;
	  exit( EXIT_FAILURE );
  }
  maskFile.close();
  itk::ImageIOBase::Pointer imageIO =
		  itk::ImageIOFactory::CreateImageIO(args.inputName.c_str(),
					itk::ImageIOFactory::ReadMode);
  imageIO->SetFileName( args.inputName );
  imageIO->ReadImageInformation();


  const int inputDimension = imageIO->GetNumberOfDimensions();
  if( args.verbose > 0 )
  {
	  cout << "Verbose output." << endl;
	  cout << "Dimensionality is " << inputDimension << "." << endl;
  }
  typedef float InputPixelType;
  switch( inputDimension )
  {
  case 2:
	  PatchAnalysis< InputPixelType, 2 >( args );
	  break;
  case 3:
	  PatchAnalysis< InputPixelType, 3 >( args );
	  break;
  case 4:
	  PatchAnalysis< InputPixelType, 4 >( args );
	  break;
  default:
	  cout << "Error: Dimension must be 2, 3, or 4." << endl;
	  exit( EXIT_FAILURE );
  }

  //printHelp();
  return 0;
  /*

  const unsigned int  Dimension = 3; // assume 3d images
  const unsigned int  NumberOfPatches = 1000; 

  typedef itk::Image< InputPixelType, Dimension >   InputImageType;
  typedef InputImageType::PointType   PointType; 


  InputImageType::Pointer PatchImage; 
  InputImageType::Pointer modality2Image; 
  InputImageType::Pointer modality2MaskImage; 

  typedef itk::ImageFileReader< InputImageType > ReaderType;
  ReaderType::Pointer  inputImageReader     = ReaderType::New();
  ReaderType::Pointer  maskImageReader      = ReaderType::New(); 
  ReaderType::Pointer  modality2ImageReader = ReaderType::New(); 
  ReaderType::Pointer  modality2MaskReader  = ReaderType::New(); 

  const unsigned int  VolumeOfPatches   = pow(double( SizeOfPatches ), 
      static_cast< int > (Dimension) ); //49; //343; // illegal: pow(SizeOfPatches, Dimension);  
  double TargetPercentVarianceExplained = atof( argv[ 6 ] ); 
  const char * modality2Filename        = argv[7]; 
  const char * modality2MaskName        = argv[8]; 

  inputImageReader->SetFileName( inputFilename );
  inputImageReader->Update();
  maskImageReader->SetFileName( maskFilename ); 
  maskImageReader->Update(); 
  modality2ImageReader->SetFileName( modality2Filename ); 
  modality2ImageReader->Update(); 
  modality2MaskReader->SetFileName( modality2MaskName ); 
  modality2MaskReader->Update(); 
  
  InputImage = inputImageReader->GetOutput(); 
  MaskImage  = maskImageReader->GetOutput(); 
  modality2Image = modality2ImageReader->GetOutput(); 
  modality2MaskImage = modality2MaskReader->GetOutput();

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
  cout << "Attempting to find seed points. Looking for " << NumberOfPatches << 
    " points out of " << inputSize << " possible points." << endl;
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
  cout << "Iterator.Size() is " << Iterator.Size() << endl;
  cout << "IndicesWithinSphere.size() is " << IndicesWithinSphere.size() << endl;

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
  const unsigned int vectorizedPatchMatrixRows = VectorizedPatchMatrix.rows(); 
  const unsigned int vectorizedPatchMatrixCols = VectorizedPatchMatrix.cols(); 
  typedef itk::CSVNumericObjectFileWriter< InputPixelType 
	   > CSVWriterType; 
  CSVWriterType::Pointer vectorizedPatchWriter = CSVWriterType::New(); 
  vectorizedPatchWriter->SetFileName("vectorizedPatches.csv"); 
  vectorizedPatchWriter->SetInput(&VectorizedPatchMatrix); 
  vectorizedPatchWriter->Update(); 
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
  int SumOfMaskImage = int( StatisticsFilter->GetSum() ); 
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
  cout << "PatchesForAllPointsWithinMask is " << PatchesForAllPointsWithinMask.rows() << "x" <<
    PatchesForAllPointsWithinMask.columns() << "..." << endl;
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
  typedef ScalarInterpolatorType::Pointer InterpPointer;
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
  int SamplePatchNumber = 1000; 
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
  int FixedIndex = 1; // reorient to second eigenvector--first is constant 
  int MovingIndex = 5;
  int NumberOfPaddingVoxels = 2; 
  int RadiusOfPatch = SizeOfPatches; 
  typedef itk::NeighborhoodIterator< InputImageType > NeighborhoodIteratorType;
  radius.Fill( RadiusOfPatch );
  InputImageType::RegionType SphereRegion;
  InputImageType::IndexType   BeginningOfSphereRegion;
  InputImageType::SizeType    SizeOfSphereRegion;
  typedef itk::CovariantVector<RealType, Dimension>                               GradientPixelType;
  typedef itk::Image<GradientPixelType, Dimension>                                GradientImageType;
  typedef itk::SmartPointer<GradientImageType>                                    GradientImagePointer;
  typedef itk::GradientRecursiveGaussianImageFilter<InputImageType, GradientImageType> GradientImageFilterType;
  typedef GradientImageFilterType::Pointer                               GradientImageFilterPointer;
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
  vnl_vector< InputPixelType > RotatedVector = MovingVector; 
  RotatedVector.fill( 0.0 ); 

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
  RotatedVector = 
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
  
  ReorientedEigvec = ConvertVectorToSpatialImage< InputImageType, 
		   InputImageType, double > (RotatedVector, EigvecMaskImage); 

  RotationWriter->SetInput(ReorientedEigvec); 
  RotationWriter->SetFileName("Rotated.nii.gz"); 
  RotationWriter->Update(); 


  vnl_matrix< InputPixelType > ReorientedPatches = 
    PatchesForAllPointsWithinMask ; 
  ReorientedPatches.fill( 0.0 );
  // allocate vars to be used in the loop
  vnl_vector< InputPixelType > MovingPatchVector = 
    SignificantPatchEigenvectors.get_column( 1 ) ;
  InputImageType::Pointer MovingPatchImage = InputImageType::New(); 
  // reorient all patches to second eigenvector
  for( int jj = 0; jj < PatchesForAllPointsWithinMask.columns(); jj++)
  {
//    int MovingPatchIndex = jj;
    MovingPatchVector = PatchesForAllPointsWithinMask.get_column( jj ); // MovingPatchIndex ); 
    MovingPatchImage = 
      ConvertVectorToSpatialImage< InputImageType, InputImageType, double > (
	  MovingPatchVector, EigvecMaskImage ); 
    GradientImageFilterPointer MovingPatchGradientFilter = GradientImageFilterType::New(); 
    MovingPatchGradientFilter->SetInput( MovingPatchImage ); 
    MovingPatchGradientFilter->SetSigma( GradientSigma ); 
    MovingPatchGradientFilter->Update(); 
    GradientImageType::Pointer MovingPatchGradientImage = 
      MovingPatchGradientFilter->GetOutput(); 
    NeighborhoodIteratorType MovingPatchIterator( radius, MovingPatchImage, SphereRegion ); 
    vnl_vector< InputPixelType > MovingPatchVectorReoriented; 
    interp1->SetInputImage( MovingPatchImage ); 
    MovingPatchVectorReoriented = 
      ReorientPatchToReferenceFrame< Dimension, InputPixelType, InputImageType, 
      GradientImageType, InterpPointer > (
	  FixedIterator, 
	  MovingPatchIterator, 
	  EigvecMaskImage, 
	  IndicesWithinSphere, 
	  Weights, 
	  FixedGradientImage, 
	  MovingPatchGradientImage, 
	  Dimension, 
	  interp1
	  ); 
    ReorientedPatches.set_column( jj, MovingPatchVectorReoriented );
    if( (jj % 1000) == 0 )
    {
      cout << "We're on index " << jj << " out of " << 
	PatchesForAllPointsWithinMask.columns() << "." << endl;
    }
  }
  cout << "Done reorienting." << endl;

  // sample reoriented patches to compute eigvecs
  int ReorientedPatchIter = 0; 
  int ReorientedPatchIndex = 0;

  vnl_matrix< InputPixelType > ReorientedPatchSamples( NumberOfPatches, ReorientedPatches.rows() ) ;
  // switch rows and columns for SVD...FIXME i think this is right
  ReorientedPatchSamples.fill( 0 ); 
  while( ReorientedPatchIter < NumberOfPatches )
  {
    ReorientedPatchIndex = rand() % ReorientedPatches.columns() ; 
    ReorientedPatchSamples.set_row( ReorientedPatchIter,  
	ReorientedPatches.get_column( ReorientedPatchIndex ) ); 
    // for svd, each row is one observation--i know it's weird...FIXME check this. 
    ReorientedPatchIter++ ; 
  }
  cout << "This is updated." << endl; 
  cout << "Got " << ReorientedPatchIter << " samples." << endl; 
  cout << "ReorientedPatchSamples is " << ReorientedPatchSamples.rows() << 
    "x" << ReorientedPatchSamples.columns() << endl;; 
  vnl_svd< InputPixelType > SVDOfReorientedPatches( ReorientedPatchSamples );
  cout << "Calculated SVD of reoriented patches." << endl;


  vnl_matrix< InputPixelType > ReorientedPatchEigenvectors = 
    SVDOfReorientedPatches.V(); 
  SumOfEigenvalues = 0.0; 
  for( int i = 0; i < SVDOfReorientedPatches.rank(); i++)
  {
    SumOfEigenvalues += SVDOfReorientedPatches.W( i, i ); 
  }
  PartialSumOfEigenvalues = 0.0; 
  PercentVarianceExplained = 0.0; 
  i = 0; 
  while( PercentVarianceExplained < TargetPercentVarianceExplained && 
      i < svd.rank() )
  {
    PartialSumOfEigenvalues += SVDOfReorientedPatches.W( i, i ); 
    PercentVarianceExplained = PartialSumOfEigenvalues / 
      SumOfEigenvalues ; 
    i++; 
  }
  int NumberOfEigenvectorsForReorientedPatches = i; 
  cout << "It took " << NumberOfEigenvectorsForReorientedPatches << 
    " eigenvectors to reach " << TargetPercentVarianceExplained * 100 << 
    "% variance explained for reoriented patches." << endl;
  
  vnl_matrix< InputPixelType > SignificantReorientedPatchEigenvectors ; 
  SignificantReorientedPatchEigenvectors = 
    ReorientedPatchEigenvectors.get_n_columns(0, NumberOfEigenvectorsForReorientedPatches );

  // write out eigenvectors 
  for ( unsigned int ii = 0; ii < NumberOfEigenvectorsForReorientedPatches; ii++)
  {
    vnl_vector< InputPixelType > EigvecAsVector =
                ReorientedPatchEigenvectors.get_column( ii );
    string ImageIndex;
    ostringstream convert;
    convert << ii;
    ImageIndex = convert.str();
    EigvecWriter->SetInput( ConvertVectorToSpatialImage< InputImageType,
        InputImageType, double >( EigvecAsVector,
          EigvecMaskImage) );
    string EigvecFileName = "ReorientedEigvec" + ImageIndex + ".nii.gz" ;
    EigvecWriter->SetFileName(EigvecFileName);
    EigvecWriter->Update();
  }



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

 



  // do same for reoriented eigenvectors

  cout << "Computing regression for reoriented eigenvectors." << endl;
  vnl_matrix< InputPixelType >
    ReorientedEigenvectorCoefficients( NumberOfEigenvectorsForReorientedPatches, SumOfMaskImage );
  ReorientedEigenvectorCoefficients.fill( 0 );
  vnl_svd< InputPixelType > ReorientedRegressionSVD(SignificantReorientedPatchEigenvectors);
//  EigenvectorCoefficients =  RegressionSVD.solve(PatchesForAllPointsWithinMask); 
//  not feasible for large matrices
  for( int i = 0; i < SumOfMaskImage; ++i)
  {
    vnl_vector< InputPixelType > ReorientedPatchOfInterest =
      ReorientedPatches.get_column(i);
    vnl_vector< InputPixelType > x( NumberOfEigenvectorsForReorientedPatches );
    x.fill( 0 );
    x = ReorientedRegressionSVD.solve(ReorientedPatchOfInterest);
    ReorientedEigenvectorCoefficients.set_column(i, x);
  }
  vnl_matrix< InputPixelType > ReconstructedReorientedPatches = 
    SignificantReorientedPatchEigenvectors * ReorientedEigenvectorCoefficients;
  vnl_matrix< InputPixelType > ReorientedError = ReconstructedReorientedPatches - ReorientedPatches; 
  vnl_vector< InputPixelType > ReorientedPercentError(ReorientedError.columns() );
  for( int i = 0; i < ReorientedError.columns(); ++i)
  {
    ReorientedPercentError(i) = ReorientedError.get_column(i).two_norm() / 
      (ReorientedPatches.get_column(i).two_norm() + 1e-10);
  }
  cout << "Average percent error for reoriented patches is " << ReorientedPercentError.mean() * 100 << "%, with max of " <<
    ReorientedPercentError.max_value() * 100 << "%." <<  endl;
  cout << "ReorientedEigenvectorCoefficients is " << ReorientedEigenvectorCoefficients.rows() << "x" <<
    ReorientedEigenvectorCoefficients.columns() << "." << endl;

  InputImageType::Pointer ReorientedConvertedImage;
  for( int i = 0; i < ReorientedEigenvectorCoefficients.rows(); ++i)
  {
    vnl_vector< InputPixelType > ReorientedRegressionCoefficient =
      ReorientedEigenvectorCoefficients.get_row(i);
    ReorientedConvertedImage = ConvertVectorToSpatialImage<InputImageType,InputImageType,double>(
      ReorientedRegressionCoefficient, MaskImage);
    typedef itk::ImageFileWriter< InputImageType > ImageWriterType;
    ImageWriterType::Pointer  ReorientedCoefficientImageWriter = ImageWriterType::New();
    ReorientedCoefficientImageWriter->SetInput( ReorientedConvertedImage );
    string ImageIndex;
    ostringstream convert;
    convert << i;
    ImageIndex = convert.str();
    string ReorientedCoefficientImageFileName = "ReorientedCoeffsOut" + ImageIndex + ".nii.gz";
    ReorientedCoefficientImageWriter->SetFileName( ReorientedCoefficientImageFileName );
    ReorientedCoefficientImageWriter->Update();
  }






  /* Predict modality B from modality A: 
   * I don't yet have masks for ASL images, 
   * so for now I'm just predicting T1 from T1 
   * to make sure the machinery works.  */
// InputImageType::Pointer modality2Image; 
//  InputImageType::Pointer modality2Mask; 
//  modality2Image = InputImage; // FIXME will change to ASL once i have masks
//  modality2Mask  = MaskImage; // ditto
  // Count number of nonzero voxels in modality2Image
  // WARNING: ASSUMES MASK IS BINARY!!!
  /*
  StatisticsFilter->SetInput( modality2MaskImage ); 
  StatisticsFilter->Update( ); 
  int sumOfModality2Mask = int( StatisticsFilter->GetSum( ) ); 
  cout << "Number of voxels to be predicted: " << sumOfModality2Mask << "." << endl;
  InputImageType::IndexType modality2Index; 
  std::vector< InputImageType::IndexType > allIndicesInModality2Image( sumOfModality2Mask );
  std::vector< PointType > centerPointsOfModality2ImagePixels( sumOfModality2Mask ); 
 
  vnl_vector < InputPixelType > vectorizedModality2Image( sumOfModality2Mask ); 
  vectorizedModality2Image.fill( 0 ); 
  ImageIteratorType modality2MaskIterator( modality2MaskImage, 
      modality2MaskImage->GetLargestPossibleRegion() );
  ImageIteratorType modality2ImageIterator( modality2Image, 
      modality2Image->GetLargestPossibleRegion() ); 
  int modality2MaskCounter = 0; 
  for( modality2MaskIterator.GoToBegin(), modality2ImageIterator.GoToBegin(); 
      !modality2MaskIterator.IsAtEnd(); 
      ++modality2MaskIterator, ++modality2ImageIterator)
  {
    if(modality2MaskIterator.Get() > 0 )
    {
      PointType IndexAsPoint; 
      vectorizedModality2Image( modality2MaskCounter ) = 
	modality2ImageIterator.Get(); 
      modality2Image->TransformIndexToPhysicalPoint( 
	    modality2MaskIterator.GetIndex(), IndexAsPoint ); 
      centerPointsOfModality2ImagePixels[ modality2MaskCounter ] = 
	IndexAsPoint; 
      allIndicesInModality2Image[ modality2MaskCounter ] = 
	modality2MaskIterator.GetIndex(); 
      modality2MaskCounter++; 
    }
  }
  cout << "Number of points is: " << modality2MaskCounter << endl;
  //reconstruct modality 2
  // resample image from modality 1 and put in a matrix
  InterpPointer interpolatorForModality1 = ScalarInterpolatorType::New(); 
  InputImageType::Pointer reorientedEigenvectorCoefficientImage; 
  vnl_matrix< InputPixelType > reorientedEigenvectorCoefficientsResampledToModality2Image( 
      ReorientedEigenvectorCoefficients.rows(), 
      centerPointsOfModality2ImagePixels.size() );
  reorientedEigenvectorCoefficientsResampledToModality2Image.fill( 0 );
  for( int ii = 0; ii < ReorientedEigenvectorCoefficients.rows(); ii++)
  {
    vnl_vector< InputPixelType >  reorientedEigenvectorCoefficientVector = 
      ReorientedEigenvectorCoefficients.get_row( ii );

    reorientedEigenvectorCoefficientImage = ConvertVectorToSpatialImage<
      InputImageType, InputImageType, double > (
	  reorientedEigenvectorCoefficientVector, MaskImage );
    interpolatorForModality1->SetInputImage( reorientedEigenvectorCoefficientImage ); 
    for( int jj = 0; jj < centerPointsOfModality2ImagePixels.size(); jj++ )
    {
      reorientedEigenvectorCoefficientsResampledToModality2Image( ii, jj ) = 
	interpolatorForModality1->Evaluate( centerPointsOfModality2ImagePixels[ jj ] );
    }


  }

  cout << "reorientedEigenvectorCoefficientsResampledToModality2Image is " << 
    reorientedEigenvectorCoefficientsResampledToModality2Image.rows() << "x" << 
    reorientedEigenvectorCoefficientsResampledToModality2Image.columns() << "." << endl;
  
  vnl_svd < InputPixelType > reorientedEigenvectorCoefficientSVD( 
      reorientedEigenvectorCoefficientsResampledToModality2Image.transpose() ); // because of funny dimensionality
  vnl_vector< InputPixelType > coefficientsForPredictingModality2( sumOfModality2Mask ); 
  coefficientsForPredictingModality2.fill( 0 ); 
  coefficientsForPredictingModality2 = 
    reorientedEigenvectorCoefficientSVD.solve( vectorizedModality2Image ); 
  vnl_vector< InputPixelType > reconstructedModality2( sumOfModality2Mask ); 
  reconstructedModality2 = ReorientedEigenvectorCoefficients.transpose() * 
    coefficientsForPredictingModality2;
  // calculate correlation coeff of reconstruction
  vnl_vector< InputPixelType > centeredVectorizedModality2Image = 
    vectorizedModality2Image - vectorizedModality2Image.mean(); 
  vnl_vector< InputPixelType > centeredReconstructedModality2 = 
    reconstructedModality2 - reconstructedModality2.mean();
  double correlationBetweenActualAndReconstructedImageModality2 = 
    inner_product( centeredVectorizedModality2Image, centeredReconstructedModality2 ) / 
    (centeredVectorizedModality2Image.two_norm() * centeredReconstructedModality2.two_norm() ); 
  cout << "CorrCoef is " << correlationBetweenActualAndReconstructedImageModality2 << "." << endl;


  return 0;
  */
}
