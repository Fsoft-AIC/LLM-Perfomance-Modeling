CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra
SRC = insert_hook_folder.cpp
FOLDER = /home/clang-llvm/kernels
SAVE_FOLDER = /home/clang-llvm/kernels-modified
EXECUTABLE = insert_hook_folder
COMMAND = insert-mem-hook

all: clean $(EXECUTABLE) run

$(EXECUTABLE): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(EXECUTABLE)
	mkdir -p $(SAVE_FOLDER)
	./$(EXECUTABLE) $(COMMAND) $(FOLDER) $(SAVE_FOLDER)

clean:
	rm -f $(EXECUTABLE)

.PHONY: all run clean
