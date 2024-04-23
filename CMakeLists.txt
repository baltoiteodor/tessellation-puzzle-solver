cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_COMPILER "C:/MinGW/bin/g++.exe")
set(CMAKE_C_COMPILER "C:/MinGW/bin/gcc.exe")
project(tpsC__)
set(CMAKE_CXX_STANDARD 17)


add_subdirectory(pybind11)

#set(SOURCE_FILES
#        DLXCPPLinker.cpp
#        dlx.cpp
#        NodeMatrix.cpp
#)

pybind11_add_module(DLXCPP DLXCPPLinker.cpp)

#add_executable(tpsC__ main.cpp)
