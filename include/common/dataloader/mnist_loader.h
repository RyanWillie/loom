#pragma once
#include <string>
#include <vector>

class MNISTLoader {
  public:
    MNISTLoader();
    void load_data(const std::string& path);
};
