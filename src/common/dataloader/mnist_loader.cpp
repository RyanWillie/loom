#include "common/dataloader/mnist_loader.h"

#include <iostream>

MNISTLoader::MNISTLoader() {
    // Constructor
}

void MNISTLoader::load_data(const std::string& path) {
    std::cout << "Loading MNIST data from " << path << std::endl;
    // TODO: Implement binary reading
}
