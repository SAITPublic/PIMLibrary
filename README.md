# FIMLibrary

FIM Runtime Library and Tools

# Setup Contribution Environment
## Docker env

All Prequisits for FIMLibrary build and testing are installed in docker image. For more info refer [Dockefile](https://github.sec.samsung.net/FIM/FIMLibrary/blob/develop/docker/Dockerfile.FimLibrary)
```
docker/docker-fim.sh <image name> <directory>

image name : docker image to be used ( SAIT-Korea : fim-tf2:rocm3.0-python3)
directory  : (optional) Directory to be mapped inside container. default your home directory is mapped
```

## MIOpen Setup ( Optional ) 
MIOpen Setup is required only if you work on modifying MIOpen code.

### Install PreRequisits
```
git clone -b roc-3.0.x git@github.sec.samsung.net:FIM/MIOpen.git
cd MIOpen
sudo cmake -P install_deps.cmake --prefix /opt/rocm
```

### Build And Install MIOpen
```
#install MIOpen
cd MIOpen
mkdir build
cd build
CXX=/opt/rocm/bin/hcc cmake -DMIOPEN_BACKEND=HIP -DCMAKE_PREFIX_PATH="/opt/rocm" -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
make -j8
sudo make install
```

# How to build
## Using Script
```
./scripts <build> <options>
<build>
all : uninstall, cmake, make, install
uninstall : removes all FIMLibrary so and binaries from rocm path
cmake : Does cmake
make : build FIMLibrary
install : installs to rocm path

<options>
-o <relative path> : Relative path from current directory where build folder is created
-d [optional] : if mentioned, Debug mode will be enabled
-m [optional] : if mentioned, MIOpen Apps will be compiled
-t [optional] : if mentioned, Target mode build will be enabled
```

## Using Commands
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=<Build Type> -DCMAKE_INSTALL_PREFIX=/opt/rocm -DMIOPEN_APPS=<option> ..
<build Type> : Release, Debug
<option>     : ON, OFF
make -j8
sudo make install
```

# Note
- Build in release mode if you want to install all Lib and Simulator to /opt/rocm/
- Logs will be generated only in Debug mode in /tmp/ directory

# Testing

## How to run FimIntegrationTests

### Run all Tests
```
./build/examples/FimIntegrationTests
```

[Optional]
### List all available Tests
``./build/examples/FimIntegrationTests --gtest_list_tests``
### Run Single Test
`` ./build/examples/FimIntegrationTests --gtest_filter_test=<Test from List>``

## How to Run MIOpen Tests
For MIOpenTests to be added to FimIntegration test, -m option need to be enabled during FIMBuild
```
./build/examples/FimIntegrationTests --gtest_filter_test=MIOpenIntegrationTests.*
```

## How To Run TF apps
```
export PYTHONPATH=/opt/rocm/lib

cd examples
python3 tf_custom_op/<test_file>
```

# Profiling of FIM Library
 Profiler has been developed for Profiling FIM Library
 
## Pre requisites
1. FIMLibrary in debug mode
   FIM Library need to be build in debug mode for generating debug logs for profiling. Logs will be generated in /tmp/ folder
2. Generate MIOpen Logs [Optional]
   MIOpen logs need to be generated for adding MIOpen Level log information in Profiler.
   ``export MIOPEN_ENABLE_LOGGING=1``
3. rocProfiler logs [Optional]
   For adding GPU profiling data
   
## Profiler Usage
For more details about usage, refer [Profiler](https://github.sec.samsung.net/FIM/FIMLibrary/tree/develop/tools/profiler)
   


