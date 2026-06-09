cmake_minimum_required(VERSION 3.16)

project(traffic_surveillance)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(thirdParty/imgui)

find_package(ncnn CONFIG REQUIRED)

find_package(Vulkan REQUIRED)


find_package(SFML COMPONENTS Graphics Window System)

file(GLOB_RECURSE MY_SOURCES CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(traffic_surveillance ${MY_SOURCES})

target_include_directories(traffic_surveillance 
		PUBLIC 
		"${CMAKE_CURRENT_SOURCE_DIR}/include"
		"${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/imgui"
)

target_compile_definitions("${CMAKE_PROJECT_NAME}" PUBLIC DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/")

target_link_libraries(traffic_surveillance
		PRIVATE
		${OpenCV_LIBS}
		Vulkan::Vulkan
		ncnn
		SFML::System 
		SFML::Window
		SFML::Graphics 
		imgui
)
