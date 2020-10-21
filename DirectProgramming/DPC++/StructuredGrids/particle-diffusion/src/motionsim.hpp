//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// motionsim.hpp: Header file for motionsim.cpp,
// motionsim_kernel.cpp, and utils.cpp
//
// Constant expressions, includes, and prototypes needed by the application.
//

// Random Number Generation (RNG) Distribution parameters
constexpr float alpha = 0.0f;   // Mean
constexpr float sigma = 0.03f;  // Standard Deviation

#if _WIN32 || _WIN64
#define WINDOWS 1
#endif  // _WIN32 || _WIN64
// unistd.h not available on windows platforms
#if !WINDOWS
#include <unistd.h>
#endif  // !WINDOWS

#include <CL/sycl.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"
// For backwards compatibility with MKL-Beta09
#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#include "oneapi/rng.hpp"
#else  // __has_include("oneapi/mkl.hpp")
#include <mkl.h>
#include "mkl_sycl.hpp"
#endif  // __has_include("oneapi/mkl.hpp")

void ParticleMotion(sycl::queue&, const int, float*, float*, float*, float*,
                    size_t*, const size_t, const size_t, const size_t,
                    const size_t, const float);
void ParticleMotionWithSnapshotPrinting(sycl::queue&, const int, float*, float*,
                                        float*, float*, size_t*, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const float,
                                        const unsigned int);
void CPUParticleMotion(const int, float*, float*, float*, float*, size_t*,
                       const size_t, const size_t, const size_t, const size_t,
                       const float);
void CPUParticleMotionWithSnapshotPrinting(const int, float*, float*, float*,
                                           float*, size_t*, const size_t,
                                           const size_t, const size_t,
                                           const size_t, const float,
                                           const unsigned int);
void Usage();
int IsNum(const char*);
bool ValidateDeviceComputation(const size_t*, const size_t*, const size_t,
                               const size_t);
bool CompareMatrices(const size_t*, const size_t*, const size_t);
int ParseArgs(const int, char* [], size_t*, size_t*, size_t*, int*,
              unsigned int*, unsigned int*, unsigned int*);
int ParseArgsWindows(int, char* [], size_t*, size_t*, size_t*, int*,
                     unsigned int*, unsigned int*, unsigned int*);
void PrintGrids(const size_t*, const size_t*, const size_t, const unsigned int,
                const unsigned int);
void PrintValidationResults(const size_t*, const size_t*, const size_t,
                            const size_t, const unsigned int,
                            const unsigned int);
void CheckVslError(int);
void ClearScreen(int);

// This function prints a vector
template <typename T>
void PrintVector(const T* vector, const size_t n) {
  std::cout << "\n";
  for (size_t i = 0; i < n; ++i) {
    std::cout << vector[i] << " ";
  }
  std::cout << "\n";
}

// This function prints a 2D matrix
template <typename T>
void PrintMatrix(const T** matrix, const size_t size_X, const size_t size_Y) {
  std::cout << "\n";
  for (size_t i = 0; i < size_X; ++i) {
    for (size_t j = 0; j < size_Y; ++j) {
      std::cout << std::setw(3) << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}

// This function prints a 1D vector as a matrix
template <typename T>
void PrintVectorAsMatrix(const T* vector, const size_t size_X,
                         const size_t size_Y) {
  std::cout << "\n";
  for (size_t j = 0; j < size_X; ++j) {
    for (size_t i = 0; i < size_Y; ++i) {
      std::cout << std::setw(3) << vector[j * size_Y + i] << " ";
    }
    std::cout << "\n";
  }
}

// This function prints a 1D vector as a matrix within DPC++ kernel
template <typename T>
void PrintVectorAsMatrixKernel(const T* vector, const size_t size_X,
                               const size_t size_Y) {
  std::cout << "\n";
  for (size_t j = 0; j < size_X; ++j) {
    for (size_t i = 0; i < size_Y; ++i) {
      std::cout << std::setw(3) << vector[j * size_Y + i] << " ";
    }
    std::cout << "\n";
  }
}
