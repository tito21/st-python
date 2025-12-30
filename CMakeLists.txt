cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)


# Generate utils.c from utils.pyx
add_custom_command(
  OUTPUT utils.c
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/utils.c from ${CMAKE_CURRENT_SOURCE_DIR}/src/st/utils.pyx"
  COMMAND Python::Interpreter -m cython
          -a
          "${CMAKE_CURRENT_SOURCE_DIR}/src/st/utils.pyx" --output-file utils.c
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/st/utils.pyx
  VERBATIM)

python_add_library(utils MODULE utils.c WITH_SOABI)

install(TARGETS utils DESTINATION st)


# Get NumPy include directory from the same Python used to build:
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NumPy_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Generate cbezier.c from cbezier.pyx
add_custom_command(
  OUTPUT cbezier.c
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/cbezier.c from ${CMAKE_CURRENT_SOURCE_DIR}/src/st/cbezier.pyx"
  COMMAND Python::Interpreter -m cython
          -a
          "${CMAKE_CURRENT_SOURCE_DIR}/src/st/cbezier.pyx" --output-file cbezier.c
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/st/cbezier.pyx
  VERBATIM)

python_add_library(cbezier MODULE cbezier.c WITH_SOABI)

target_include_directories(
  cbezier
  PRIVATE
    "${NumPy_INCLUDE_DIR}"
    ${Python_INCLUDE_DIRS}
)

install(TARGETS cbezier DESTINATION st)


# Generate simplify_tract.c from simplify_tract.pyx
add_custom_command(
  OUTPUT simplify_tract.c
  COMMENT
    "Making ${CMAKE_CURRENT_BINARY_DIR}/simplify_tract.c from ${CMAKE_CURRENT_SOURCE_DIR}/src/st/simplify_tract.pyx"
  COMMAND Python::Interpreter -m cython
          -a
          "${CMAKE_CURRENT_SOURCE_DIR}/src/st/simplify_tract.pyx" --output-file simplify_tract.c
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/st/simplify_tract.pyx
  VERBATIM)

python_add_library(simplify_tract MODULE simplify_tract.c WITH_SOABI)

target_include_directories(
  simplify_tract
  PRIVATE
    "${NumPy_INCLUDE_DIR}"
    ${Python_INCLUDE_DIRS}
)


install(TARGETS simplify_tract DESTINATION st)