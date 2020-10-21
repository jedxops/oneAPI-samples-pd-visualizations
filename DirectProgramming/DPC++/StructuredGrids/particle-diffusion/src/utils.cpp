//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// utils: Intel® oneAPI DPC++ utility functions
// used in motionsim.cpp and motionsim_kernel.cpp
//

#include "motionsim.hpp"
using namespace std;

// This function displays correct usage and parameters
void Usage() {
  cout << "Particle Diffusion DPC++ code sample help message:\n";
#if !WINDOWS
  cout << "\nUsage: ./<binary_name> <flags>"
       << "\n-------------------------------------------------------------"
       << "\n|Flag | Variable name         | Range      | Default value  |"
       << "\n|-----|-----------------------|------------|----------------|"
       << "\n|-i   | n_iterations          | [1, inf]   | [default=10000]|"
       << "\n|-p   | n_particles           | [1, inf]   | [default=256]  |"
       << "\n|-g   | grid_size             | [1, inf]   | [default=22]   |"
       << "\n|-r   | seed                  | [-inf, inf]| [default=777]  |"
       << "\n|-c   | cpu_flag              | [0, 1]     | [default=0]    |"
       << "\n|-s   | output_snapshots_flag | [0, 1]     | [default=1]    |"
       << "\n|-o   | grid_output_flag      | [0, 1]     | [default=1]    |"
       << "\n-------------------------------------------------------------\n\n";
#else   // WINDOWS
  cout << "\nUsage: ";
  cout << "./<binary_name> <Number of Iterations> <Number of Particles> "
       << "<Size of Square Grid> <Seed for RNG> <1/0 Flag for CPU Comparison> "
       << "<1/0 Flag for Snapshot Output> <1/0 Flag for Grid Output>"
       << "\n--------------------------------------------------------"
       << "\n|Argument name           | Range      | Default value  |"
       << "\n|------------------------|------------|----------------|"
       << "\n|Number of Iterations    | [1, inf]   | [default=10000]|"
       << "\n|Number of Particles     | [1, inf]   | [default=256]  |"
       << "\n|Size of Square Grid     | [1, inf]   | [default=22]   |"
       << "\n|Seed for RNG            | [-inf, inf]| [default=777]  |"
       << "\n|Flag for CPU comparison | [0, 1]     | [default=0]    |"
       << "\n|Flag for snapshot output| [0, 1]     | [default=1]    |"
       << "\n|Flag for Grid Output    | [0, 1]     | [default=1]    |"
       << "\n--------------------------------------------------------\n\n";
#endif  // WINDOWS
}

// Returns true for numeric strings, used for argument parsing
int IsNum(const char* str) {
  for (int i = 0; i < strlen(str); ++i)
    if (!isdigit(str[i])) return 1;
  return 0;
}

// Examines two matricies and returns true if they are equivalent.
bool ValidateDeviceComputation(const size_t* grid_device,
                               const size_t* grid_cpu, const size_t grid_size,
                               const size_t planes) {
  for (int c = 0; c < grid_size; ++c)
    for (int r = 0; r < grid_size; ++r)
      for (int d = 0; d < planes; ++d)
        if (grid_device[r + grid_size * c + grid_size * grid_size * d] !=
            grid_cpu[r + grid_size * c + grid_size * grid_size * d])
          return false;
  return true;
}

// Returns false for NxN matrices which do not contain the same values
bool CompareMatrices(const size_t* grid1, const size_t* grid2,
                     const size_t grid_size) {
  for (int c = 0; c < grid_size; ++c)
    for (int r = 0; r < grid_size; ++r)
      if (grid1[r + c * grid_size] != grid2[r + c * grid_size]) return false;
  return true;
}

#if !WINDOWS
// Command line argument parser
int ParseArgs(const int argc, char* argv[], size_t* n_iterations,
              size_t* n_particles, size_t* grid_size, int* seed,
              unsigned int* cpu_flag, unsigned int* output_snapshots_flag,
              unsigned int* grid_output_flag) {
  int retv = 0;
  int negative_seed = 0;
  int option;
  // Parse user-specified parameters
  while ((option = getopt(argc, argv, "i:p:g:r:c:s:o:h")) != -1 && !retv) {
    if (optarg) {
      if (option == 'r' && optarg[0] == '-') negative_seed = 1;
      if (!negative_seed) retv = IsNum(optarg);
      if (retv) goto usage_label;
    }
    switch (option) {
      case 'i':
        *n_iterations = stoi(optarg);
        break;
      case 'p':
        *n_particles = stoi(optarg);
        break;
      case 'g':
        *grid_size = stoi(optarg);
        break;
      case 'r':
        *seed = stoi(optarg);
        break;
      case 'c':
        *cpu_flag = stoul(optarg);
        break;
      case 's':
        *output_snapshots_flag = stoul(optarg);
        break;
      case 'o':
        *grid_output_flag = stoul(optarg);
        break;
      case 'h':
      case ':':
      case '?':
      default:
      usage_label : {
        retv = 1;
        break;
      }
    }
  }
  if ((*cpu_flag != 1 && *cpu_flag != 0) ||
      (*grid_output_flag != 1 && *grid_output_flag != 0) ||
      (*output_snapshots_flag != 1 && *output_snapshots_flag != 0) ||
      (*n_iterations == 0))
    retv = 1;
  if (retv) Usage();
  return retv;
}  // End of function ParseArgs()
#else   // WINDOWS
// Windows command line argument parser
int ParseArgsWindows(int argc, char* argv[], size_t* n_iterations,
                     size_t* n_particles, size_t* grid_size, int* seed,
                     unsigned int* cpu_flag, unsigned int* output_snaphots_flag,
                     unsigned int* grid_output_flag) {
  int retv = 0;
  // Parse user-specified parameters
  try {
    for (int i = 1; i < argc; ++i)
      if (stoi(argv[i]) < 0 && i != 4) retv = 1;
    *n_iterations = stoi(argv[1]);
    *n_particles = stoi(argv[2]);
    *grid_size = stoi(argv[3]);
    *seed = stoi(argv[4]);
    *cpu_flag = stoul(argv[5]);
    *output_snapshots_flag = stoul(argv[6]);
    *grid_output_flag = stoul(argv[7]);
  } catch (...) {
    retv = 1;
  }
  if ((*cpu_flag != 1 && *cpu_flag != 0) ||
      (*grid_output_flag != 1 && *grid_output_flag != 0) ||
      (*output_snapshots_flag != 1 && *output_snapshots_flag != 0) ||
      (*n_iterations == 0))
    retv = 1;
  if (retv) Usage();
  return retv;
}
#endif  // WINDOWS

// Prints the 3D grids holding the simulation results
void PrintGrids(const size_t* grid, const size_t* grid_cpu,
                const size_t grid_size, const unsigned int cpu_flag,
                const unsigned int grid_output_flag) {
  // Index variables for 3rd dimension of grid
  size_t layer = 0;
  const size_t gs2 = grid_size * grid_size;

  // Displays final grid only if grid small
  if (grid_size <= 45 && grid_output_flag) {
    cout << "\n**********************************************************\n";
    cout << "*                           DEVICE                       *\n";
    cout << "**********************************************************\n";

    cout << "\n **************** PARTICLE ACCUMULATION: ****************\n";

    // Counter 1 layer of grid (0 * grid_size * grid_size)
    layer = 0;
    PrintVectorAsMatrix<size_t>(&grid[layer], grid_size, grid_size);

    cout << "\n ***************** FINAL SNAPSHOT: *****************\n";

    // Counter 2 layer of grid (1 * grid_size * grid_size)
    layer = gs2;
    PrintVectorAsMatrix<size_t>(&grid[layer], grid_size, grid_size);
    // Calculate and print the total number of particles in the snapshot
    unsigned int psum = 0;
    for (unsigned int i = layer; i < layer + gs2; ++i) psum += grid[i];
    cout << "Number of particles inside snapshot: " << psum << "\n";

    cout << "\n ************* NUMBER OF PARTICLE ENTRIES: ************* \n";

    // Counter 3 layer of grid (2 * grid_size * grid_size)
    layer = gs2 + gs2;
    PrintVectorAsMatrix<size_t>(&grid[layer], grid_size, grid_size);

    cout << "**********************************************************\n";
    cout << "*                        END DEVICE                      *\n";
    cout << "**********************************************************\n\n\n";
  }

  if (cpu_flag) {
    // Displays final grid on cpu only if grid small
    if (grid_size <= 45 && grid_output_flag) {
      cout << "\n";
      cout << "**********************************************************\n";
      cout << "*                           CPU                          *\n";
      cout << "**********************************************************\n";

      cout << "\n **************** PARTICLE ACCUMULATION: ****************\n";

      // Counter 1 layer of grid (0 * grid_size * grid_size)
      layer = 0;
      PrintVectorAsMatrix<size_t>(&grid_cpu[layer], grid_size, grid_size);

      cout << "\n ***************** FINAL SNAPSHOT: *****************\n";

      // Counter 2 layer of grid (1 * grid_size * grid_size)
      layer = gs2;
      PrintVectorAsMatrix<size_t>(&grid_cpu[layer], grid_size, grid_size);
      // Calculate and print the total number of particles in the snapshot
      unsigned int psum = 0;
      for (unsigned int i = layer; i < layer + gs2; ++i) psum += grid_cpu[i];
      cout << "Number of particles inside snapshot: " << psum << "\n";

      cout << "\n ************* NUMBER OF PARTICLE ENTRIES: ************* \n";

      // Counter 3 layer of grid (2 * grid_size * grid_size)
      layer = gs2 + gs2;
      PrintVectorAsMatrix<size_t>(&grid_cpu[layer], grid_size, grid_size);

      cout << "**********************************************************\n";
      cout << "*                          END CPU                       *\n";
      cout << "**********************************************************\n";
      cout << "\n\n";
    }
  }
}  // End of function PrintGrids()

// Compares the matrices generated by the CPU and device and prints results.
void PrintValidationResults(const size_t* grid, const size_t* grid_cpu,
                            const size_t grid_size, const size_t planes,
                            const unsigned int cpu_flag,
                            const unsigned int grid_output_flag) {
  // Index variables for 3rd dimension of grid
  size_t layer = 0;
  const size_t gs2 = grid_size * grid_size;
  bool retv = false;

  /* ********** Counter 1: device v.s cpu comparison ********** */

  // Counter 1 layer of grid (0 * grid_size * grid_size)
  layer = 0;
  retv = CompareMatrices(&grid[layer], &grid_cpu[layer], grid_size);

  if (!retv)
    cout << "MISMATCH. Device Counter 1 != CPU Counter 1\n";
  else
    cout << "MATCH. Device Counter 1 = CPU Counter 1\n";

  /* ********** Counter 2: device v.s cpu comparison ********** */

  // Counter 2 layer of grid (1 * grid_size * grid_size)
  layer = gs2;
  retv = CompareMatrices(&grid[layer], &grid_cpu[layer], grid_size);

  if (!retv)
    cout << "MISMATCH. Device Counter 2 != CPU Counter 2\n";
  else
    cout << "MATCH. Device Counter 2 = CPU Counter 2\n";

  /* ********** Counter 3: device v.s cpu comparison ********** */

  // Counter 3 layer of grid (2 * grid_size * grid_size)
  layer = gs2 + gs2;
  retv = CompareMatrices(&grid[layer], &grid_cpu[layer], grid_size);

  if (!retv)
    cout << "MISMATCH. Device Counter 3 != CPU Counter 3\n";
  else
    cout << "MATCH. Device Counter 3 = CPU Counter 3\n";

  retv = ValidateDeviceComputation(grid, grid_cpu, grid_size, planes);

  if (!retv)
    cout << "\nError. CPU computation does not match that of the device.\n";
  else
    cout << "\nSuccess.\n";
}  // End of function PrintValidationResults()

// Error checks mkl RNG function during CPU random number generation
void CheckVslError(int err_code) {
  if (err_code != VSL_ERROR_OK && err_code != VSL_STATUS_OK) {
    cout << "Encountered VSL Error during random number generation on CPU. "
            "Exiting.\n";
    exit(EXIT_FAILURE);
  }
}

// Outputs n*10+1 newlines (clears screen and flushes output buffer)
void ClearScreen(int n) {
  for (int i = 0; i < n; ++i) cout << "\n\n\n\n\n\n\n\n\n\n";
  cout << endl;
}
