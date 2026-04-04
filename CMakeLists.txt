cmake_minimum_required(VERSION 3.15...3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

find_package(Cython MODULE REQUIRED VERSION 3.0)
include(UseCython)

cython_transpile("./src/st/utils.pyx" LANGUAGE C CYTHON_ARGS -a OUTPUT_VARIABLE utils_c)
cython_transpile("./src/st/cbezier.pyx" LANGUAGE C CYTHON_ARGS -a OUTPUT_VARIABLE bezier_c)
cython_transpile("./src/st/simplify_tract.pyx" LANGUAGE C CYTHON_ARGS -a OUTPUT_VARIABLE simplify_tract_c)
cython_transpile("./src/st/img_brush.pyx" LANGUAGE C CYTHON_ARGS -a OUTPUT_VARIABLE img_brush_c)

python_add_library(utils MODULE "${utils_c}" WITH_SOABI)
python_add_library(cbezier MODULE "${bezier_c}" WITH_SOABI)
python_add_library(simplify_tract MODULE "${simplify_tract_c}" WITH_SOABI)
python_add_library(img_brush MODULE "${img_brush_c}" WITH_SOABI)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
target_link_libraries(img_brush PRIVATE Python::NumPy)
target_link_libraries(simplify_tract PRIVATE Python::NumPy)
target_link_libraries(cbezier PRIVATE Python::NumPy)
install(TARGETS utils cbezier simplify_tract img_brush DESTINATION st)
