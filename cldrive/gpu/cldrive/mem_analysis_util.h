#include "boost/filesystem.hpp"
#include "boost/filesystem/fstream.hpp"
#include "labm8/cpp/logging.h"

#include <json/json.h>
#include <sstream>
#include <iostream>
#include <cmath>

namespace gpu {
namespace cldrive {
namespace mem_analysis {
  // generate the memory analysis file path from the source file path and the memory analysis directory
  boost::filesystem::path getMemAnalysisFilePath(std::string sourcePath_, std::string memAnalysisDir_);

  // check if the memory analysis file exists
  bool isMemAnalysisFileExists(std::string filePathToCheck_, std::string memAnalysisDir_);

  // get the memory analysis information from the memory analysis file
  std::map<int,int> getMemAnalysisInfo(boost::filesystem::path memFilePath, int gsize, int lsize);
  std::map<int,int> getMemAnalysisInfo(std::string memFilePathStr, int gsize, int lsize);
  std::map<int, int> getMemAnalysisInfo(std::string sourceFile_, std::string memAnalysisDir_, int gsize, int lsize);
}
}
}
