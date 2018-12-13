project(speech-tools)
include_directories(
  ${CMAKE_SOURCE_DIR}/vad/webrtcvad
  )

aux_source_directory(. DIR_LIB_SRCS)

# add predefine macros
ADD_DEFINITIONS(-DWEBRTC_POSIX)

add_library(webrtc-signal-processing ${DIR_LIB_SRCS})

# add link lib(-lpthread -lm)
target_link_libraries(webrtc-signal-processing -lpthread)

# MESSAGE("This is signal processing " ${DIR_LIB_SRCS})
