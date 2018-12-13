project(speech-tools)
include_directories(
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/vad/webrtcvad
  )

# subdirectory need to be compile.
add_subdirectory(webrtc/common_audio/signal_processing)
add_subdirectory(webrtc/common_audio/vad)

# DIR_LIB_SRCS is get from it's father
aux_source_directory(. DIR_LIB_SRCS)

# generate a target
add_executable(vad-webrtc ${DIR_LIB_SRCS})

# add link lib
target_link_libraries(vad-webrtc kaldi-base kaldi-matrix kaldi-util kaldi-feat webrtc-signal-processing webrtc-libvad)

# MESSAGE("This is apply-vad " ${DIR_LIB_SRCS})
