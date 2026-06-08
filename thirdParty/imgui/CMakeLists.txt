project(imgui LANGUAGES CXX)

file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS
    "*.cpp"
)


# find_package(glfw3 CONFIG REQUIRED)

find_package(OpenGL REQUIRED)

find_package(SFML COMPONENTS Graphics Window System)

add_library(imgui STATIC ${MY_SOURCES})

target_link_libraries(imgui
  PRIVATE 
		# glfw
		OpenGL::GL
)

