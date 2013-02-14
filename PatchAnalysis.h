template < class  TInputImageType, class TRealType > 
class PatchAnalysisObject
{
  public:
    PatchAnalysisObject( TInputImageType, TInputImageType,  int );// image, mask, radius of patch
    void GenerateMaskImageFromPatch( 
	std::vector< unsigned int > &, 
	const unsigned int &, 
	const unsigned int &); 

};
