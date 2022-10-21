# PIMLibrary

PIM Runtime library and tools to execute computations on PIM hardware.  
Platforms supported : HIP , OpenCL

# Setup Contribution Environment
## Docker env

All Prerequisites for PIMLibrary build and testing are installed in docker image. For more info refer [Dockefile](https://github.sec.samsung.net/PIM/PIMLibrary/blob/develop/docker/Dockerfile.PimLibrary)
```
cd PIMLibrary/docker
./docker-fim.sh <image-name> <directory>

image name : docker image to be used ( pim-rocm4.0:tf2.3-dev )
directory  : (optional) Directory to be mapped inside container. default your home directory is mapped
```
# How to build
## Using Script
using build.sh script.
```
./scripts/build.sh <build> <options>
<build>
all : uninstall, cmake, make, install
uninstall : removes all PIMLibrary so and binaries from rocm path
cmake : Does cmake
make : build PIMLibrary
install : installs to rocm path

<options>
-o <relative path> : Relative path from current directory where build folder is created
-d --debug  [optional] : if mentioned, Debug mode will be enabled.
-t --target [optional] TARGET : represent which target hardware PIMLibrary is built for.  {default : AMD device as target device}
supported targets : AMD and NVIDIA.  
-e --emulator [optional]: enables the execution in emulator mode (where the computation for PIM are imitated using a simulator) for functionality check and debugging.  
```
if build is in PIMLibrary directory, to build from scratch using the script, below command can be used:  
### TARGET mode  (AMD as target)
```
./scripts/build.sh all -o . -t amd
```
### Emulator mode  (AMD as target)
```
./scripts/build.sh all -o . -t amd -e
```
### TARGET mode  (NVIDIA as target)
```
./scripts/build.sh all -o . -t nvidia
```
### Emulator mode  (NVIDIA as target)
```
./scripts/build.sh all -o . -t nvidia -e
```
## Using Commands
defaults :  
~BUILD_TYPE = RELEASE  
~EXECUTION MODE = TARGET  
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=<Build Type> -DEMULATOR=<option> -DCMAKE_INSTALL_PREFIX=$ROCM_PATH ..
<build Type> : Release, Debug
<option>     : ON, OFF
make -j8
sudo make install
```

# Note
- Build in release mode if you want to install all Lib and Simulator to ROCM_PATH
- Logs will be generated only in Debug mode in /tmp/ directory

## DLOG Generation
For generating debug logs,
1. Set log level
``` 
export PIM_LOG_LEVEL=<severity> 
severity
0 : INFO
1 : WARNING
2 : ERROR
```
2. Compile PIMLibrary in Debug mode. (cmake, build and install required) : use all and -d option in build script
3. Execute application. Debug logs will be dumped in /tmp/ directory with date and time as filename.

# Testing

## How to run PimIntegrationTests

### Run all Tests
### HIP
```
./build/examples/PimIntegrationTests
```
### OpenCL
```
./build/examples/OpenCLPimIntegrationTests
```

### Run Single Test
- List all available Tests
append --gtest_list_tests to the execute command.  
``./build/examples/<executable-binary> --gtest_list_tests``  

 - Run the Test  
append --gtest_filter flag to execute command.  
`` ./build/examples/<executable-binary> --gtest_filter=<Test from List>``

-executable-binary : PimIntegrationTests (HIP) / OpenCLPimIntegrationTests (OpenCL)

# NOTE  
- to generate traces (dump of all instructions PIM HW is supposed to run in order) of the kernel execution , run the test in emulator mode (-e) with debugging flag (-d) enabled.
- the traces will be generated in test_vectors/dump/operation_name folder with 2 .dat files :   
  - fmtd16.dat (traces before coleascing of threads)  
  - fmtd32.dat (traces after coleascing of threads)  
 format of trace: ``index , block_id , thread_id , command , address``

# Profiling of PIM Library
## PIM Library profiler
Profiler has been developed for Profiling PIM Library

### Pre requisites
1. PIMLibrary in debug mode
   PIM Library need to be build in debug mode for generating debug logs for profiling. Logs will be generated in /tmp/ folder

### Profiler Usage
For more details about usage, refer [Profiler](https://github.sec.samsung.net/PIM/PIMLibrary/tree/develop/tools/profiler)

# Documentation
## How to generate Doxygen documentation
### Prerquisites
doxygen and graphviz packages need to be installed.
```
sudo apt-get install doxygen
sudo apt-get install graphviz
```

### Documentation Generation
`doxygen Doxyfile`

Documentation will be generated in Doc/Doxygen folder

