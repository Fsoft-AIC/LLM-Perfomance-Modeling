
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include "clang/AST/ParentMapContext.h"

#include <fstream>
#include <filesystem>
#include <map>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory MyToolCategory("my-tool options");
static cl::opt<std::string> saveFolderPath(
      "save_dir",
      cl::desc("Folder path (default: file's parent directory/modified)"),
      cl::init(""));

Rewriter Rewrite;
std::map<std::string, int> argMap;

std::string mapToJson(const std::map<std::string, int>& myMap) {
    std::string jsonString = "{";

    for (auto it = myMap.begin(); it != myMap.end(); ++it) {
        if (it != myMap.begin()) {
            jsonString += ",";
        }
        jsonString += "\"" + it->first + "\":" + std::to_string(it->second);
    }

    jsonString += "}";

    return jsonString;
}


// Calculate the distance between two source locations
unsigned calculateSourceLocationDistance(ASTContext &Context, SourceLocation StartLoc, SourceLocation EndLoc) {
    const SourceManager &SM = Context.getSourceManager();

    // Ensure that the source locations are in the same file
    if (SM.getFileID(StartLoc) != SM.getFileID(EndLoc)) {
        return 0; // Return 0 to indicate they are in different files.
    }

    // Get the character offsets of the source locations
    unsigned StartOffset = SM.getFileOffset(StartLoc);
    unsigned EndOffset = SM.getFileOffset(EndLoc);
    
    // Calculate the distance as the absolute difference in character offsets
    return (StartOffset < EndOffset) ? (EndOffset - StartOffset + 1) : (StartOffset - EndOffset + 1);
}


class InsertHookClassVisitor
  : public RecursiveASTVisitor<InsertHookClassVisitor> {
public:
  explicit InsertHookClassVisitor(ASTContext *Context, std::map<std::string, int>& argMap)
    : argMap(argMap), Context(Context) {}

  bool VisitArraySubscriptExpr(ArraySubscriptExpr *ASE) {
        if (ASE->getBase() && ASE->getIdx()) {
          Expr *Idx = ASE->getIdx();
          SourceLocation startLoc = Idx->getBeginLoc();
          SourceLocation endLoc = ASE->getEndLoc();
          LangOptions LO = Rewrite.getLangOpts();
          
          // Get the index of the argument
          if (argMap.find(exprToString(ASE->getBase())) == argMap.end()) {
            argMap[exprToString(ASE->getBase())] = argMap.size();
          }
          int argIndex = argMap[exprToString(ASE->getBase())];
          std::string argIndexStr = std::to_string(argIndex);
          std::string hookCall = "hook(" + argIndexStr + ", " + exprToString(ASE->getIdx()) + ")";

          // calculate the length of the original expression
          unsigned orgLength = calculateSourceLocationDistance(*Context, startLoc, endLoc) - 1;
          // Insert a call to your memory hook function
          Rewrite.ReplaceText(startLoc, orgLength, hookCall);
        }
        return true;
  }

private:
  ASTContext *Context;
  std::map<std::string, int>& argMap;

  // Helper function to convert an expression to a string
  std::string exprToString(Expr *E) {
      std::string Str;
      llvm::raw_string_ostream Stream(Str);
      E->printPretty(Stream, nullptr, PrintingPolicy(Context->getLangOpts()));
      return Stream.str();
  }
};

class GetKernelArgsASTVisitor : public RecursiveASTVisitor<GetKernelArgsASTVisitor> {
public:
    explicit GetKernelArgsASTVisitor(std::map<std::string, int>& argMap) : argMap(argMap) {}

    bool VisitFunctionDecl(clang::FunctionDecl *FD) {
        // Only function definitions (with bodies), not declarations and only OpenCL kernel functions
        if (FD->hasBody() && FD->hasAttr<clang::OpenCLKernelAttr>()) {
            // Process the function declaration.
            for (auto param : FD->parameters()) {
                std::string argName = param->getNameAsString();
                std::string argType = param->getOriginalType().getAsString();
                int argIndex = param->getFunctionScopeIndex();
                argMap[argName] = argIndex;
            }
        }
        return true;
    }

private:
    std::map<std::string, int>& argMap;
};

class InsertHookClassConsumer : public clang::ASTConsumer {
public:
  explicit InsertHookClassConsumer(ASTContext *Context)
    : getKernelArgsVisitor(argMap), insertHookClassVisitor(Context, argMap) {}

  virtual void HandleTranslationUnit(clang::ASTContext &Context) {
    // Traverse the AST to collect kernel arguments
    argMap.clear();
    getKernelArgsVisitor.TraverseDecl(Context.getTranslationUnitDecl());
    Rewrite = Rewriter(Context.getSourceManager(), Context.getLangOpts());
    insertHookClassVisitor.TraverseDecl(Context.getTranslationUnitDecl());
    
  }
private:
  GetKernelArgsASTVisitor getKernelArgsVisitor;
  InsertHookClassVisitor insertHookClassVisitor;
};

// Given a FileID, get the file name
std::string getModifiedFilePath(const SourceManager &SM, FileID FID, std::string saveFolderPath) {
    const FileEntry *FE = SM.getFileEntryForID(FID);
    if (FE) {
      std::filesystem::path orgFilePath = FE->getName().str();
      std::filesystem::path modifiedFilePath = saveFolderPath / orgFilePath.filename();
      return modifiedFilePath.string();
    }
    return "Unknown";
}

class InsertHookClassAction : public clang::ASTFrontendAction {
public:
  InsertHookClassAction(std::string saveFolderPath)
    : saveFolderPath(saveFolderPath) {}

  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
    clang::CompilerInstance &Compiler, llvm::StringRef InFile) {
    return std::make_unique<InsertHookClassConsumer>(&Compiler.getASTContext());
  }
protected:
  virtual void EndSourceFileAction() {
    FileID ID = Rewrite.getSourceMgr().getMainFileID();

    // Get the modified source code from the rewriter
    std::string modifiedCode;
    llvm::raw_string_ostream modifiedCodeStream(modifiedCode);

    // map argMap to json string, then add it as a comment
    modifiedCodeStream << "//" + mapToJson(argMap) + "\n";

    modifiedCodeStream << "int hook(int argId, int id) {\n";
    modifiedCodeStream << "\tprintf(\"%d,%d\\n\", argId, id);\n";
    modifiedCodeStream << "\treturn id;\n";
    modifiedCodeStream << "}\n";
    Rewrite.getEditBuffer(ID).write(modifiedCodeStream);
    modifiedCodeStream.flush();
    
    // Write the modified code to a file
    std::string fileName = getModifiedFilePath(Rewrite.getSourceMgr(), ID, saveFolderPath); 
    if (fileName == "Unknown") {
      llvm::errs() << "Failed to get the modified file name\n";
      return;
    }
    llvm::outs() << "Writing modified code to file " << fileName << "\n";
    std::ofstream outFile(fileName);
    outFile << modifiedCode;
    outFile.close();
  }
private:
  std::string saveFolderPath;
};

std::unique_ptr<FrontendActionFactory> myNewFrontendActionFactory(std::string options) {
  class SimpleFrontendActionFactory : public FrontendActionFactory {
   public:
    SimpleFrontendActionFactory(std::string saveFolderPath) : saveFolderPath(saveFolderPath) {}

    std::unique_ptr<FrontendAction> create() override {
      return std::make_unique<InsertHookClassAction>(saveFolderPath);
    }

   private:
    std::string saveFolderPath;
  };
 
  return std::unique_ptr<FrontendActionFactory>(
      new SimpleFrontendActionFactory(options));
}


int main(int argc, const char **argv) {
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectedParser) {
    // Fail gracefully for unsupported options.
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser& OptionsParser = ExpectedParser.get();

  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  if (saveFolderPath.empty()) {
    // Get the parent directory of the first file
    std::string filePath = OptionsParser.getSourcePathList()[0];
    std::filesystem::path savePath = std::filesystem::path(filePath).parent_path() / "modified";
    saveFolderPath = savePath.string();
  }
  llvm::outs() << "saveFolderPath: " << saveFolderPath << "\n";
  return Tool.run(myNewFrontendActionFactory(saveFolderPath).get());
}
