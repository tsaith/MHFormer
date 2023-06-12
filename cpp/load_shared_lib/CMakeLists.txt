cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_bg_matting)


if (UNIX) # For Ubuntu
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu")
    #set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/libtorch")
else () # For Win10
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:\\opt\\opencv\\build")
    #set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:\\opt\\libtorch")
endif (UNIX)

#find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS} ${TORCH_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS})
set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl") 

#set(HEADER libbg_matting.h) 
#set(SRC bg_matting.cpp background_matting.cpp) 

add_executable(test_bg_matting detect.cpp bg_matting.hpp)

set_target_properties(test_bg_matting PROPERTIES
    CXX_STANDARD 14)

target_link_libraries(test_bg_matting ${OpenCV_LIBS})
#target_link_libraries(test_bg_matting ${TORCH_LIBRARIES})