# Feature Selector

I made this for a school project so the parser expects the file's delimiter to be " ", as well as no header. Change the formatter code in *include/data.h* to suit your use case. 

**Dependencies / Borrowed Code**
- [vincentlaucsb/csv-parser](https://github.com/vincentlaucsb/csv-parser) for parsing text files
- [fmtlib/fmt](http://github.com/fmtlib/fmt) for simple printing and formatting

**Requirements:**
- A C++ compiler that supports C++20 features
- CMake or some other build system

**Sample Installation and Build Commands Using CMake**
```
git clone https://github.com/mooogy/feature_selector.git
cd feature_selector

cmake -B build
cmake --build build
./build/feature_selector  
```
