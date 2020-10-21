//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// main: Intel® oneAPI DPC++ Language Basics Using a Monte Carlo
// Simulation (main function)
//
// This code sample will implement a simple example of a Monte Carlo
// simulation of the diffusion of water molecules in tissue.
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide
// and search based on relevant terms noted in the comments.
//
// DPC++ material used in this code sample:
//
// Basic structures of DPC++:
//   DPC++ Queues (including device selectors and exception handlers)
//   DPC++ Buffers and accessors (communicate data between the host and the
//   device) DPC++ Kernels (including parallel_for function and range<2>
//   objects) API-based programming: Use oneMKL to generate random numbers
//   (DPC++) DPC++ atomic operations for synchronization
//

#include "motionsim.hpp"
namespace oneapi {}
using namespace oneapi;
using namespace sycl;
using namespace std;

// Main Function
int main(int argc, char* argv[]) {
  // Set command line arguments to their default values
  size_t n_iterations = 10000;
  size_t n_particles = 256;
  size_t grid_size = 22;
  int seed = 777;
  unsigned int cpu_flag = 0;
  unsigned int output_snapshots_flag = 1;
  unsigned int grid_output_flag = 1;

  cout << "\n";
  if (argc == 1)
    cout << "**Running with default parameters**\n\n";
  else {
    int rc = 0;
// Detect OS type and read in command line arguments
#if !WINDOWS
    rc = ParseArgs(argc, argv, &n_iterations, &n_particles, &grid_size, &seed,
                   &cpu_flag, &output_snapshots_flag, &grid_output_flag);
#elif WINDOWS  // WINDOWS
    rc = ParseArgsWindows(argc, argv, &n_iterations, &n_particles, &grid_size,
                          &seed, &cpu_flag, &output_snapshots_flag,
                          &grid_output_flag);
#else          // WINDOWS
    cout << "Error. Failed to detect operating system. Exiting.\n";
    return 1;
#endif         // WINDOWS
    if (rc != 0) return 1;
  }  // End else

  // Allocate and initialize arrays
  //

  // Stores X and Y position of particles in the cell grid
  float* particle_X = new float[n_particles];
  float* particle_Y = new float[n_particles];
  // Total number of motion events
  const size_t n_moves = n_particles * n_iterations;
  // Declare vectors to store random values for X and Y directions
  float* random_X = new float[n_moves];
  float* random_Y = new float[n_moves];
  // Grid center
  const float center = grid_size / 2;
  // Initialize the particle starting positions to the grid center
  for (size_t i = 0; i < n_particles; ++i) {
    particle_X[i] = center;
    particle_Y[i] = center;
  }

  /*  Each of the folowing counters represent a separate plane in the grid
  variable (described below).
  Each plane is of size: grid_size * grid_size. There are 3 planes.

  _________________________________________________________________________
  |COUNTER                                                           PLANE|
  |-----------------------------------------------------------------------|
  |counter 1: Total particle accumulation counts for each cell            |
  |           throughout the entire simulation. Includes             z = 0|
  |           particles that remained in, as well as particles            |
  |           that have returned to each cell.                            |
  |                                                                       |
  |counter 2: Current total number of particles in each cell.        z = 1|
  |           Does not increase if a particle remained in a cell.         |
  |           An instantaneous snapshot of the grid of cells.             |
  |                                                                       |
  |counter 3: Total number of particle entries into each cell.       z = 2|
  |_______________________________________________________________________|

  The 3D matrix is implemented as a 1D matrix to improve efficiency.

  For any given index j with coordinates (x, y, z) in a N x M x D (Row x
  Column x Depth) 3D matrix, the same index j in an equivalently sized 1D
  matrix is given by the following formula:

  j = y + M * x + M * M * z                                              */

  // Size of 3rd dimension of 3D grid (3D matrix). 3 counters => 3 planes
  const size_t planes = 3;
  // Stores a grid of cells, initialized to zero.
  size_t* grid = new size_t[grid_size * grid_size * planes]();
  // Cell radius = 0.5*(grid spacing)
  const float radius = 0.5f;

  // Create a device queue using default or host/device selectors
  default_selector device_selector;
  // Create a device queue using DPC++ class queue
  queue q(device_selector, dpc_common::exception_handler);

  // Start timers
  dpc_common::TimeInterval t_offload;
  // Call device simulation function
  if (output_snapshots_flag)
    ParticleMotionWithSnapshotPrinting(
        q, seed, particle_X, particle_Y, random_X, random_Y, grid, grid_size,
        planes, n_particles, n_iterations, radius, output_snapshots_flag);
  else
    ParticleMotion(q, seed, particle_X, particle_Y, random_X, random_Y, grid,
                   grid_size, planes, n_particles, n_iterations, radius);
  q.wait_and_throw();
  auto device_time = t_offload.Elapsed();
  // End timers

  cout << "\nDevice Offload time: " << device_time << " s\n\n";

  size_t* grid_cpu;
  // If user wants to perform cpu computation, for comparison with device
  // Off by default
  if (cpu_flag) {
    // Re-initialize arrays
    for (size_t i = 0; i < n_particles; ++i) {
      particle_X[i] = center;
      particle_Y[i] = center;
    }

    // Use Math Kernel Library (MKL) VSL Gaussian function for RNG with
    // mean of alpha and standard deviation of sigma
    //

    VSLStreamStatePtr stream;
    int vsl_retv = vslNewStream(&stream, VSL_BRNG_PHILOX4X32X10, seed);
    CheckVslError(vsl_retv);

    vsl_retv = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n_moves,
                             random_X, alpha, sigma);
    CheckVslError(vsl_retv);

    vsl_retv = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, n_moves,
                             random_Y, alpha, sigma);
    CheckVslError(vsl_retv);

    grid_cpu = new size_t[grid_size * grid_size * planes]();

    // Start timers
    dpc_common::TimeInterval t_offload_cpu;
    // Call CPU simulation function
    if (output_snapshots_flag)
      CPUParticleMotionWithSnapshotPrinting(
          seed, particle_X, particle_Y, random_X, random_Y, grid_cpu, grid_size,
          planes, n_particles, n_iterations, radius, output_snapshots_flag);
    else
      CPUParticleMotion(seed, particle_X, particle_Y, random_X, random_Y,
                        grid_cpu, grid_size, planes, n_particles, n_iterations,
                        radius);
    auto cpu_time = t_offload_cpu.Elapsed();
    // End timers

    cout << "\nCPU Offload time: " << cpu_time << " s\n\n";
  }
  if (grid_output_flag)
    PrintGrids(grid, grid_cpu, grid_size, cpu_flag, grid_output_flag);
  if (cpu_flag)
    PrintValidationResults(grid, grid_cpu, grid_size, planes, cpu_flag,
                           grid_output_flag);
  else
    cout << "Success.\n";

  // Cleanup
  if (cpu_flag) delete[] grid_cpu;
  delete[] grid;
  delete[] random_X;
  delete[] random_Y;
  delete[] particle_X;
  delete[] particle_Y;
  return 0;
}  // End of function main()
