# Compile and build
## Edit CMakeLists.txt
mkdir build
cd build
cmake .. -D CMAKE_INSTALL_PREFIX=/opt
make
sudo make install
