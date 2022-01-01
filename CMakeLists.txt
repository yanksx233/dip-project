cmake_minimum_required(VERSION 3.19)
project(dip_project)

# 设置OpenCV目录所在
set(OpenCV_DIR "D:\\Program Files\\opencv\\mingw64-build")
# 搜索OpenCV目录
find_package(OpenCV REQUIRED)
# 添加OpenCV头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(dip_fa main.cpp)

# 链接OpenCV库文件
target_link_libraries(dip_fa ${OpenCV_LIBS})

# add libs you need
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)