https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_windows.md

cmake -G Ninja -B build -S . -D CMAKE_TOOLCHAIN_FILE=C:\Users\admin\source\repos\vcpkg\scripts\buildsystems\vcpkg.cmake -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="-g"
