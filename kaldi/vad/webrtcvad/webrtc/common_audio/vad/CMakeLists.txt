project(speech-tools)
include_directories(
  ${CMAKE_SOURCE_DIR}/vad/webrtcvad
  )

aux_source_directory(. DIR_LIB_SRCS)
add_library(webrtc-libvad ${DIR_LIB_SRCS})
