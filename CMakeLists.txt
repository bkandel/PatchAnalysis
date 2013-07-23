cmake_minimum_required(VERSION 2.8)
FIND_PACKAGE(ITK)

IF(ITK_FOUND)
  INCLUDE(${ITK_USE_FILE})
ELSE(ITK_FOUND)
  MESSAGE(FATAL_ERROR 
          "ITK not found. Please set ITK_DIR.")
ENDIF(ITK_FOUND)
PROJECT(PatchAnalysisProject)
ADD_EXECUTABLE(PatchAnalysis PatchAnalysis.cxx)
ADD_EXECUTABLE(TestPatchPropogation TestPatchPropogation.cxx)
TARGET_LINK_LIBRARIES(PatchAnalysis ${ITK_LIBRARIES})
TARGET_LINK_LIBRARIES(TestPatchPropogation ${ITK_LIBRARIES})
