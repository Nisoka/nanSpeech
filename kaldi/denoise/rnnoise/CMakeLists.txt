project(speech-tools)
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/include
  )

aux_source_directory(. DIR_LIB_SRCS)

# ADD_DEFINITIONS(-fpermissive)
add_executable(denoise-rnn ${DIR_LIB_SRCS})

# add link lib
target_link_libraries(denoise-rnn -lm kaldi-base kaldi-matrix kaldi-util kaldi-feat)
