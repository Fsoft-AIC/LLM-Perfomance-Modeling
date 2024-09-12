#include <iostream>
#include <filesystem>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    std::string command = "insert-mem-hook"; // Default command
    std::string folderPath = "/home/clang-llvm/kernels"; // Default folder path
    std::string saveFolderPath = "/home/clang-llvm/kernels-modified"; // Default save folder path

    // Check if command-line arguments are provided
    if (argc >= 2) {
        command = argv[1];
    }
    if (argc >= 3) {
        folderPath = argv[2];
    }
    if (argc >= 4) {
        saveFolderPath = argv[3];
    }

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            // Get the file path
            std::string filePath = entry.path().string();

            // Build and execute the command
            std::string fullCommand = command + " " + filePath + " -save_dir=" + saveFolderPath + " --";
            int result = std::system(fullCommand.c_str());

            if (result == 0) {
                std::cout << "Command `" << fullCommand << "` executed successfully for: " << filePath << std::endl;
            } else {
                std::cerr << "Command `" << fullCommand << "` execution failed for: " << filePath << std::endl;
            }
        }
    }

    return 0;
}
