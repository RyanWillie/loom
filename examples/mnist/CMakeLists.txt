# MNIST Example
add_executable(mnist_example main.cpp)

# Link against the loom library
target_link_libraries(mnist_example PRIVATE loom)

# Set output name
set_target_properties(mnist_example PROPERTIES OUTPUT_NAME "mnist")
