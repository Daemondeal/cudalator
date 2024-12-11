.SUFFIXES:
.PHONY: run clean purge .dummy

EXE := cudalator-compiler
COMPILER_BUILD := ./build/cudalator
EXE_PATH := $(COMPILER_BUILD)/$(EXE)

# ROOT_DIR := $(shell pwd)
LIB_DIR := ./build/libraries

$(EXE_PATH): $(COMPILER_BUILD)/build.ninja .dummy
	cd $(COMPILER_BUILD) && ninja

.dummy:
	@# Used for forcing a rebuild always

run: $(EXE_PATH)
	$(EXE_PATH) ./data/rtl/adder.sv


$(COMPILER_BUILD)/build.ninja: CMakeLists.txt $(LIB_DIR)/.timestamp $(LIB_DIR)/spdlog.timestamp
	mkdir -p build
	cmake -B $(COMPILER_BUILD) -G Ninja

$(LIB_DIR)/.timestamp:
	@echo '### BUILDING SURELOG ###'
	@echo ''

	mkdir -p $(LIB_DIR)
	cd ./external/Surelog/ && $(MAKE)
	cd ./external/Surelog/ && $(MAKE) PREFIX=$(LIB_DIR) install
	touch $(LIB_DIR)/.timestamp

$(LIB_DIR)/spdlog.timestamp:
	@echo '### BUILDING SPDLOG ###'
	@echo ''

	mkdir -p $(LIB_DIR)
	mkdir -p ./external/spdlog/build
	cd ./external/spdlog/build && cmake .. && cmake --build .
	touch $(LIB_DIR)/spdlog.timestamp


clean:
	rm -rf $(COMPILER_BUILD)

purge:
	rm -rf build
