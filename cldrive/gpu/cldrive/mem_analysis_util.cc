#include "gpu/cldrive/mem_analysis_util.h"

namespace gpu {
namespace cldrive {
namespace mem_analysis {
  boost::filesystem::path getMemAnalysisFilePath(std::string sourcePath_, std::string memAnalysisDir_) {
    // Specify the directory path storing the memory analysis
    boost::filesystem::path memAnalysisDir(memAnalysisDir_);

    // get the base name without extension of filePathToCheck_
    boost::filesystem::path sourcePath(sourcePath_);

    // get the corresponding memory analysis file path
   return memAnalysisDir / sourcePath.filename().replace_extension(".json");
  }


  bool isMemAnalysisFileExists(std::string filePathToCheck_, std::string memAnalysisDir_) {
    // get the file path to check
    boost::filesystem::path memFileToCheck = getMemAnalysisFilePath(filePathToCheck_, memAnalysisDir_);

    return boost::filesystem::exists(memFileToCheck);
  }

  std::map<int,int> getMemAnalysisInfo(boost::filesystem::path memFilePath, int gsize, int lsize) {
    std::map<int, int> memAnalysisInfo;

    // If the file not exists, then use default memory analysis setting (empty)
    if (boost::filesystem::exists(memFilePath) == false) {
      return memAnalysisInfo;
    }

    std::ifstream jsonFile(memFilePath.string());

    // Read the JSON content into a string
    std::string jsonData((std::istreambuf_iterator<char>(jsonFile)),
                         std::istreambuf_iterator<char>());

    // Close the file
    jsonFile.close();

    // Parse the JSON data
    Json::Value root;
    Json::Reader reader;

    CHECK(reader.parse(jsonData, root)) << "Failed to parse JSON data: " << reader.getFormattedErrorMessages() << "\n";

    // Iterate through children of root and get the memory analysis information
    for (const auto& child : root.getMemberNames()) {
        const Json::Value& currentChild = root[child];
        int argId = currentChild["arg_id"].asInt();
        int arraySize = std::round(currentChild["coef"][0].asDouble() * gsize 
                              + currentChild["coef"][1].asDouble() * lsize 
                              + currentChild["coef"][2].asDouble()) + 1;
        memAnalysisInfo[argId] = arraySize;
    }

    return memAnalysisInfo;
  }

  std::map<int,int> getMemAnalysisInfo(std::string memFilePathStr, int gsize, int lsize) {
    // get the file path to check
    boost::filesystem::path memFilePath(memFilePathStr);
    return getMemAnalysisInfo(memFilePath, gsize, lsize);
  }

  std::map<int, int> getMemAnalysisInfo(std::string sourceFile_, std::string memAnalysisDir_, int gsize, int lsize) {
    // get the file path to check
    boost::filesystem::path memFilePath = getMemAnalysisFilePath(sourceFile_, memAnalysisDir_);
    return getMemAnalysisInfo(memFilePath, gsize, lsize);
  }
}
}
}
