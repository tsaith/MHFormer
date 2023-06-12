cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(detect)


if (UNIX) # For Ubuntu
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/libtorch")
else () # For Win
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:\\opt\\opencv\\build")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:\\opt\\libtorch")
endif (UNIX)

#include_directories("/home/andrew/anaconda3/include")
#link_directories("/home/andrew/anaconda3/lib")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS} ${TORCH_CXX_FLAGS})
set (CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE} -ldl") 

set(SRC main.cpp mocap_mp_proxy_linux.cpp) 

add_executable(detect main.cpp mocap_mp_proxy_linux.cpp)
target_link_libraries(detect ${OpenCV_LIBS})
target_link_libraries(detect ${TORCH_LIBRARIES})

set_property(TARGET detect PROPERTY CXX_STANDARD 17)

