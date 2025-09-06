.PHONY: all

all: ./build/build.timestamp
	cd build && make && ./run_simulation


./build/build.timestamp:
	mkdir -p build
	cd build && cmake $(CMAKE_ARGS) ..
	touch build/build.timestamp
