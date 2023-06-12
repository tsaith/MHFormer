cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bg_matting VERSION 1.0.1 DESCRIPTION "Shared library of BackgroundMattingV2.")


if (UNIX) # For Ubuntu
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/libtorch")
else () # For Win10
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:\\opt\\opencv\\build")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:\\opt\\libtorch")
endif (UNIX)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS} ${TORCH_CXX_FLAGS})

#set(HEADER libbg_matting.h) 
set(SRC libbg_matting.cpp background_matting.cpp) 

add_library(bg_matting SHARED ${SRC})

set_target_properties(bg_matting PROPERTIES
    VERSION ${PROJECT_VERSION}
    #PUBLIC_HEADER ${HEADER}
    CXX_STANDARD 14)

target_link_libraries(bg_matting ${OpenCV_LIBS})
target_link_libraries(bg_matting ${TORCH_LIBRARIES})

install(TARGETS bg_matting
    #PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})