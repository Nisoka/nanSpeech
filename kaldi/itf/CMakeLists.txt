project(speech-tools)
include_directories(${CMAKE_SOURCE_DIR})
aux_source_directory(. DIR_LIB_SRCS)
add_library(kaldi-itf ${DIR_LIB_SRCS})
