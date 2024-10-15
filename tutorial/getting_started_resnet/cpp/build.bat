cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ^
      -DBUILD_SHARED_LIBS=OFF ^
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON ^
      -DCMAKE_CONFIGURATION_TYPES=Release ^
      -A x64 ^
      -T host=x64 ^
      -DCMAKE_INSTALL_PREFIX=. ^
      -DCMAKE_PREFIX_PATH="C:/Users/budcr/source/repos/opencv/mybuild/build;C:/Users/budcr/source/repos/eigen-3.4.0" ^
      -B build ^
      -S resnet_cifar ^
      -DOpenCV_DIR="C:/Users/budcr/source/repos/opencv/mybuild/build" ^
      -G "Visual Studio 17 2022"

cmake --build "%cd%\build" --config "Release" --clean-first -- /p:CL_MPcount= /nodeReuse:False

cmake --install "%cd%\build" --config "Release"

