project(speech-tools)
include_directories(
  ${CMAKE_SOURCE_DIR}/vad/webrtcvad
  )

# aux function add all the .c file to DIR_LIB_SRCS
aux_source_directory(. DIR_LIB_SRCS)

# generate a lib
add_library(webrtc-libvad ${DIR_LIB_SRCS})

# add link lib
target_link_libraries(webrtc-libvad webrtc-signal-processing)
