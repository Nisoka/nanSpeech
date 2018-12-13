project(speech-tools)
include_directories(
  ${CMAKE_SOURCE_DIR}/vad/webrtcvad
  )

add_subdirectory(webrtc/common_audio/signal_processing)
add_subdirectory(webrtc/common_audio/vad)

aux_source_directory(. DIR_LIB_SRCS)
MESSAGE("This is apply-vad " ${DIR_LIB_SRCS})


add_executable(apply-vad ${DIR_LIB_SRCS})
target_link_libraries(apply-vad webrtc-signal-processing webrtc-libvad)

