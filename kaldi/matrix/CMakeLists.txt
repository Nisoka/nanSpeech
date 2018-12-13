project(speech-tools)
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/ATLAS_headers/include
  )

add_definitions(-DHAVE_ATLAS)
aux_source_directory(. DIR_LIB_SRCS)

add_library(kaldi-matrix ${DIR_LIB_SRCS})
target_link_libraries(kaldi-matrix kaldi-base)
