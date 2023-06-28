cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mhformer VERSION 1.0.1 DESCRIPTION "Shared library of MHFormer.")


if (UNIX) # For Ubuntu
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/libtorch")
else () # For Win11
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:\\opt\\opencv\\build")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:\\opt\\libtorch")
endif (UNIX)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenCV_CXX_FLAGS} ${TORCH_CXX_FLAGS})

set(SRC libmhformer.cpp mhformer.cpp mhformer_utils.cpp) 

add_library(mhformer SHARED ${SRC})

set_target_properties(mhformer PROPERTIES
    VERSION ${PROJECT_VERSION}
    CXX_STANDARD 17)

target_link_libraries(mhformer ${OpenCV_LIBS})
target_link_libraries(mhformer ${TORCH_LIBRARIES})

install(TARGETS mhformer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
