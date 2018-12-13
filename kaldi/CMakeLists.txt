cmake_minimum_required(VERSION 3.5)
project(speech-tools)
enable_language(CXX)
# add compile options -std=c++11

# add definitions macro
# add_definitions(-std=c++11)
# ADD_DEFINITIONS(-fpermissive)

# SET(CMAKE_CXX_COMPILER /usr/bin/g++)
# SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -std=c++11")
# add debug info
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb --std=c++11")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -std=c++11")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(
  include
  )


# set(denoise-rnn denoise-rnn.cc)
# subdirectory need to be compile.
add_subdirectory(matrix)
add_subdirectory(base)
add_subdirectory(util)
add_subdirectory(feat)
add_subdirectory(denoise/rnnoise)
add_subdirectory(vad/webrtcvad)
