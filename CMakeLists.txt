CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
project(yolov8_onnxruntime)
add_compile_options(-std=c++17 -O3 -Wall -pedantic -Wextra)

SET(ONNXRUNTIME_DIR /opt/onnxruntime)  # onnxruntime root

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  )

catkin_package(
  INCLUDE_DIRS include ${ONNXRUNTIME_DIR}/include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
   roscpp
)

# --- Configure your project files ---
# include_directories(include)  # Include your header files directory

# # Recursively collect all source files under 'src' directory
# file(GLOB_RECURSE CURR_SOURCES src/*.cpp)

# # Create the executable
# add_executable(YOLOv8CPP ${CURR_SOURCES})

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17 -O3)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_DIR}/include
)

set (${PROJECT_NAME}_CPP_SOURCES
src/nn/autobackend.cpp
src/nn/onnx_model_base.cpp 
src/utils/augment.cpp
src/utils/common.cpp
src/utils/ops.cpp
)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_CPP_SOURCES})
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${catkin_LIBRARIES} ${OpenCV_LIBS} ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)