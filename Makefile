.SUFFIXES:
.PHONY: run clean purge

SOURCES := src/*.cpp

EXE := cudalator

ROOT_DIR := $(shell pwd)
LIB_DIR := $(ROOT_DIR)/libraries

./build/$(EXE): ./build/build.ninja $(SOURCES)
	cd build && ninja

run: ./build/$(EXE)
	./build/$(EXE) ./rtl_examples/adder.sv


./build/build.ninja: CMakeLists.txt $(LIB_DIR)/.timestamp
	cmake -B build -G Ninja

$(LIB_DIR)/.timestamp:
	@echo '### BUILDING SURELOG ###'
	@echo ''

	cd ./third_party/Surelog/ && $(MAKE)
	cd ./third_party/Surelog/ && $(MAKE) PREFIX=$(LIB_DIR) install
	touch $(LIB_DIR)/.timestamp


clean:
	rm -rf build

purge:
	rm -rf build libraries
