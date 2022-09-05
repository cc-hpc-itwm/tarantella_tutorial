cmake_minimum_required(VERSION 3.12)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include (read_version)
configure_version(VERSION_FILE "${CMAKE_SOURCE_DIR}/version.py")

project(tarantella-tutorial VERSION ${TNT_VERSION})

find_package(Sphinx)
add_subdirectory(docs)

