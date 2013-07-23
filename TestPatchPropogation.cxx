//#include "PatchAnalysis.h"
//#include "PatchAnalysis.hxx"
#include "itkImage.h"
#include "itkImageFileWriter.h"
int main()
{
  typedef float PixelType; 
  const unsigned int Dimension = 2; 
  typedef itk::Image< PixelType, Dimension > ImageType; 

  ImageType::RegionType region; 
  ImageType::IndexType start; 
  start[0] = 0; 
  start[1] = 0; 

  ImageType::SizeType size; 
  size[0] = 50;
  size[1] = 50; 

  region.SetSize(size); 
  region.SetIndex(start); 
  ImageType::Pointer structuralImage =              ImageType::New(); 
  ImageType::Pointer functionalImageFromStructure = ImageType::New(); 
  ImageType::Pointer functionalImage =              ImageType::New(); 
  structuralImage->SetRegions(region); 
  structuralImage->Allocate(); 
  functionalImageFromStructure->SetRegions(region); 
  functionalImageFromStructure->Allocate(); 
  functionalImage->SetRegions(region); 
  functionalImage->Allocate(); 


  for( int i = 0; i < 30; i++ ) 
  {
    for (int j = 0; j < 3; j++)
    {
      ImageType::IndexType pixelIndex;
      pixelIndex[0] = 10 * ( j + 1 );
      pixelIndex[1] = i + 10; 
      structuralImage->SetPixel(pixelIndex, 1);
      functionalImageFromStructure->SetPixel(pixelIndex, 1); 
      functionalImage->SetPixel(pixelIndex, 1); 
    }
  }
  for (int i = 5; i < 35; i++)
  {
    ImageType::IndexType pixelIndex; 
    pixelIndex[0] = i; 
    pixelIndex[1] = 25; 
    structuralImage->SetPixel(pixelIndex, 1); 
    functionalImageFromStructure->SetPixel(pixelIndex, 2);
    functionalImage->SetPixel(pixelIndex, 2); 
  }
  for (int i = 0; i < 3; i++)
  {
    ImageType::IndexType pixelIndex; 
    pixelIndex[0] = 10 * ( i + 1); 
    pixelIndex[1] = 25; 
    structuralImage->SetPixel(pixelIndex, 1); 
    functionalImageFromStructure->SetPixel(pixelIndex, 3); 
    functionalImage->SetPixel(pixelIndex, 3); 
  }

  for (int i = 10; i < 20; i++)
  {
    ImageType::IndexType pixelIndex; 
    pixelIndex[0] = 10; 
    pixelIndex[1] = i; 
    functionalImage->SetPixel(pixelIndex, 4); 
  }
  typedef itk::ImageFileWriter< ImageType > WriterType; 
  WriterType::Pointer writer = WriterType::New(); 
  writer->SetFileName("Structural.nii.gz"); 
  writer->SetInput( structuralImage ); 
  writer->Update(); 

  writer->SetFileName("FunctionalFromStructural.nii.gz"); 
  writer->SetInput(functionalImageFromStructure); 
  writer->Update(); 

  writer->SetFileName("Functional.nii.gz"); 
  writer->SetInput( functionalImage ); 
  writer->Update(); 


  return 0; 
}
