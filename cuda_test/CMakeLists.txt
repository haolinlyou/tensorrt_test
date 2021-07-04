cmake_minimum_required(VERSION 3.0.0)
project(cmaketest 
    VERSION 0.1.0 
    LANGUAGES CUDA CXX)

# set(CMAKE_CXX_COMPILE "nvcc")


include_directories(${PROJECT_SOURCE_DIR}/include)

message("source files:" ${PROJECT_SOURCE_DIR}/src)



aux_source_directory(${PROJECT_SOURCE_DIR}/src SRCS)
# file(GLOB SRCS "${PROJECT_SOURCE_DIR}"/src)

# Find all main*.cpp files and store in list mains
# file(GLOB_RECURSE mains RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

foreach(mainfile ${SRCS})
    message("---src:" ${mainfile})
    # Get file name without directory
    get_filename_component(mainname ${mainfile} NAME_WE)
    add_executable(${mainname} ${mainfile})
endforeach()