cmake_minimum_required(VERSION 3.5)
project(speech-tools)
enable_language(CXX)
# SET(CMAKE_CXX_COMPILER /usr/bin/g++)
# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# add debug info
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(
  include
  )

# set(denoise-rnn denoise-rnn.cc)

add_subdirectory(matrix)
add_subdirectory(base)
add_subdirectory(util)
add_subdirectory(feat)
add_subdirectory(denoise/rnnoise)
add_subdirectory(vad/webrtcvad)

