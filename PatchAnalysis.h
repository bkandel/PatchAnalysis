struct ArgumentType
{
  std::string inputName;           // -i option
  std::string maskName;            // -m option
  std::string outPatchName;        // -p option
  std::string eigvecName;          // -e option
  int patchSize;                  // -s option
  double targetVarianceExplained; // -t option
  int numberOfSamplePatches;      // -n option
  int verbose;                    // -v option
  int help;                       // -h option
};

template < class ImageType >
class TPatchAnalysis
{
public:
	TPatchAnalysis( ArgumentType &, const int  );
	void SetArguments( ArgumentType & );
	void ReadInputImage( void );
	void ReadMaskImage( void );
	void GetSamplePatchLocations( void );
	void ExtractSamplePatches( void );
	void LearnEigenPatches( void );
	void WriteEigenPatches( void );
	void ExtractAllPatches( void );
	void ProjectOnEigenPatches( void );
	void WriteProjections( void );

private:
	ArgumentType args;
	typename ImageType::Pointer inputImage;
	typename ImageType::Pointer maskImage;
	vnl_matrix < int > patchSeedPoints;
	std::vector< unsigned int > indicesWithinSphere;
	vnl_matrix< typename ImageType::PixelType > vectorizedPatchMatrix;
	vnl_matrix< typename ImageType::PixelType > significantPatchEigenvectors;
	int dimension;
	vnl_matrix< typename ImageType::PixelType > patchesForAllPointsWithinMask;
	long unsigned int numberOfVoxelsWithinMask;
	vnl_matrix< typename ImageType::PixelType > eigenvectorCoefficients;
};
