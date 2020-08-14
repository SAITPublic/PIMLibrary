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

## DLOG Generation
For generating debug logs, 
1. Set log level
``` 
export FIM_LOG_LEVEL=<severity> 
severity
0 : INFO
1 : WARNING
2 : ERROR
```
2. Compile FIMLibrary in Debug mode. (cmake, build and install required) : use all and -d option in build script
3. Execute application. Debug logs will be dumped in /tmp/ directory with date and time as filename.

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
## FIM Library profiler
Profiler has been developed for Profiling FIM Library
 
### Pre requisites
1. FIMLibrary in debug mode
   FIM Library need to be build in debug mode for generating debug logs for profiling. Logs will be generated in /tmp/ folder
2. Generate MIOpen Logs [Optional]
   MIOpen logs need to be generated for adding MIOpen Level log information in Profiler.
   ``export MIOPEN_ENABLE_LOGGING=1``
3. rocProfiler logs [Optional]
   For adding GPU profiling data
   
### Profiler Usage
For more details about usage, refer [Profiler](https://github.sec.samsung.net/FIM/FIMLibrary/tree/develop/tools/profiler)
   

## ROC-profiler
ROC profiler developed by ROCm developer tools. It provides low-level performance analysis for profiling GPU compute applications.

### Availability
ROC profiler can be downloaded and installed from the repository: https://github.com/ROCm-Developer-Tools/rocprofiler

### Profiler Usage

Roc profiler provides a variety of options to gather profiling information. 
```
rocprof [-h] [--list-basic] [--list-derived] [-i <input .txt/.xml file>] [-o <output CSV file>] <app command line>

Options:
  -h - this help
  --verbose - verbose mode, dumping all base counters used in the input metrics
  --list-basic - to print the list of basic HW counters
  --list-derived - to print the list of derived metrics with formulas
  --cmd-qts <on|off> - quoting profiled cmd-line [on]

  -i <.txt|.xml file> - input file
      Input file .txt format, automatically rerun application for every pmc line:

        # Perf counters group 1
        pmc : Wavefronts VALUInsts SALUInsts SFetchInsts FlatVMemInsts LDSInsts FlatLDSInsts GDSInsts FetchSize
        # Perf counters group 2
        pmc : VALUUtilization,WriteSize L2CacheHit
        # Filter by dispatches range, GPU index and kernel names
        # supported range formats: "3:9", "3:", "3"
        range: 1 : 4
        gpu: 0 1 2 3
        kernel: simple Pass1 simpleConvolutionPass2

      Input file .xml format, for single profiling run:

        # Metrics list definition, also the form "<block-name>:<event-id>" can be used
        # All defined metrics can be found in the 'metrics.xml'
        # There are basic metrics for raw HW counters and high-level metrics for derived counters
        <metric name=SQ:4,SQ_WAVES,VFetchInsts
        ></metric>

        # Filter by dispatches range, GPU index and kernel names
        <metric
          # range formats: "3:9", "3:", "3"
          range=""
          # list of gpu indexes "0,1,2,3"
          gpu_index=""
          # list of matched sub-strings "Simple1,Conv1,SimpleConvolution"
          kernel=""
        ></metric>

  -o <output file> - output CSV file [<input file base>.csv]
    The output CSV file columns meaning in the columns order:
      Index - kernels dispatch order index
      KernelName - the dispatched kernel name
      gpu-id - GPU id the kernel was submitted to
      queue-id - the ROCm queue unique id the kernel was submitted to
      queue-index - The ROCm queue write index for the submitted AQL packet
      tid - system application thread id which submitted the kernel
      grd - the kernel's grid size
      wgr - the kernel's work group size
      lds - the kernel's LDS memory size
      scr - the kernel's scratch memory size
      vgpr - the kernel's VGPR size
      sgpr - the kernel's SGPR size
      fbar - the kernel's barriers limitation
      sig - the kernel's completion signal
      ... - The columns with the counters values per kernel dispatch
      DispatchNs/BeginNs/EndNs/CompleteNs - timestamp columns if time-stamping was enabled
      
  -d <data directory> - directory where profiler store profiling data including thread treaces [/tmp]
      The data directory is renoving autonatically if the directory is matching the temporary one, which is the default.
  -t <temporary directory> - to change the temporary directory [/tmp]
      By changing the temporary directory you can prevent removing the profiling data from /tmp or enable removing from not '/tmp' directory.

  --basenames <on|off> - to turn on/off truncating of the kernel full function names till the base ones [off]
  --timestamp <on|off> - to turn on/off the kernel dispatches timestamps, dispatch/begin/end/complete [off]
    Four kernel timestamps in nanoseconds are reported:
        DispatchNs - the time when the kernel AQL dispatch packet was written to the queue
        BeginNs - the kernel execution begin time
        EndNs - the kernel execution end time
        CompleteNs - the time when the completion signal of the AQL dispatch packet was received

  --ctx-limit <max number> - maximum number of outstanding contexts [0 - unlimited]
  --heartbeat <rate sec> - to print progress heartbeats [0 - disabled]
  --obj-tracking <on|off> - to turn on/off kernels code objects tracking [off]
    To support V3 code-object.

  --stats - generating kernel execution stats, file <output name>.stats.csv
  
  --roctx-trace - to enable rocTX application code annotation trace, "Markers and Ranges" JSON trace section.
  --sys-trace - to trace HIP/HSA APIs and GPU activity, generates stats and JSON trace chrome-tracing compatible
  --hip-trace - to trace HIP, generates API execution stats and JSON file chrome-tracing compatible
  --hsa-trace - to trace HSA, generates API execution stats and JSON file chrome-tracing compatible
  --kfd-trace - to trace KFD, generates API execution stats and JSON file chrome-tracing compatible
    Generated files: <output name>.<domain>_stats.txt <output name>.json
    Traced API list can be set by input .txt or .xml files.
    Input .txt:
      hsa: hsa_queue_create hsa_amd_memory_pool_allocate
    Input .xml:
      <trace name="HSA">
        <parameters list="hsa_queue_create, hsa_amd_memory_pool_allocate">
        </parameters>
      </trace>

  --trace-start <on|off> - to enable tracing on start [on]
  --trace-period <dealy:length:rate> - to enable trace with initial delay, with periodic sample length and rate
    Supported time formats: <number(m|s|ms|us)>
```



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


