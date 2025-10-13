all: build
	./build/main

pre_build: CMakeLists.txt 
	./build.sh

build: pre_build main.cpp
	cmake --build build

clean:build
	rm -rf build
