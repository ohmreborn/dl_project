all: build
	cmake --build build

build: CMakeLists.txt
	./build.sh

run:build
	./build/main

clean:build
	rm -rf build
