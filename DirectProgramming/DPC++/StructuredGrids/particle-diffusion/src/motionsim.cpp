//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// motionsim: Intel® oneAPI DPC++ Language Basics Using a Monte Carlo
// Simulation
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

// This function distributes simulation work
void CPUParticleMotion(const int seed, float* particle_X, float* particle_Y,
                       float* random_X, float* random_Y, size_t* grid,
                       const size_t grid_size, const size_t planes,
                       const size_t n_particles, const size_t n_iterations,
                       const float radius) {
  // Grid size squared
  const size_t gs2 = grid_size * grid_size;

  cout << "Running on: CPU\n";
  cout << "Number of iterations: " << n_iterations << "\n";
  cout << "Number of particles: " << n_particles << "\n";
  cout << "Size of the grid: " << grid_size << "\n";
  cout << "Random number seed: " << seed << "\n";

  // Array of flags for each particle.
  // Each array initialized to false/zero with new array zero initializer.
  // True when particle is found to be in a cell
  bool* inside_cell = new bool[n_particles]();
  // Operations flags
  bool* increment_C1 = new bool[n_particles]();
  bool* increment_C2 = new bool[n_particles]();
  bool* increment_C3 = new bool[n_particles]();
  bool* decrement_C2_for_previous_cell = new bool[n_particles]();
  bool* update_coordinates = new bool[n_particles]();
  // Coordinates of the last known cell this particle resided in.
  // Initialized in motion simulation algorithm below and presumably not used
  // before initialization. Thus, avoid spending computation time initializing
  unsigned int* prev_known_cell_coordinate_X = new unsigned int[n_particles];
  unsigned int* prev_known_cell_coordinate_Y = new unsigned int[n_particles];

  // Motion simulation algorithm
  // --Start iterations--
  // Each iteration:
  //    1. Updates the position of all particles
  //    2. Checks if particle is inside a cell or not
  //    3. Updates counters in cells array (grid)
  //

  // All n_particles particles need to each be displaced once per iteration
  // to match device's algorithm
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    for (size_t p = 0; p < n_particles; ++p) {
      // Set the displacements to the random numbers
      float displacement_X = random_X[iter * n_particles + p];
      float displacement_Y = random_Y[iter * n_particles + p];
      // Displace particles
      particle_X[p] += displacement_X;
      particle_Y[p] += displacement_Y;
      // Compute distances from particle position to grid point i.e.,
      // the particle's distance from center of cell. Subtract the
      // integer value from floating point value to get just the
      // decimal portion. Use this value to later determine if the
      // particle is inside or outside of the cell
      float dX = sycl::abs(particle_X[p] - sycl::round(particle_X[p]));
      float dY = sycl::abs(particle_Y[p] - sycl::round(particle_Y[p]));
      /* Grid point indices closest the particle, defined by the following:
      ------------------------------------------------------------------
      |               Condition               |         Result         |
      |---------------------------------------|------------------------|
      |particle_X + 0.5 >= ceiling(particle_X)|iX = ceiling(particle_X)|
      |---------------------------------------|------------------------|
      |particle_Y + 0.5 >= ceiling(particle_Y)|iY = ceiling(particle_Y)|
      |---------------------------------------|------------------------|
      |particle_X + 0.5 < ceiling(particle_X) |iX = floor(particle_X)  |
      |---------------------------------------|------------------------|
      |particle_Y + 0.5 < ceiling(particle_Y) |iY = floor(particle_Y)  |
      ------------------------------------------------------------------  */
      int iX = sycl::floor(particle_X[p] + 0.5);
      int iY = sycl::floor(particle_Y[p] + 0.5);

      /* There are 5 cases when considering particle movement about the
         grid.

         All 5 cases are distinct from one another; i.e., any particle's
         motion falls under one and only one of the following cases:

           Case 1: Particle moves from outside cell to inside cell
                   --Increment counters 1-3
                   --Turn on inside_cell flag
                   --Store the coordinates of the
                     particle's new cell location

           Case 2: Particle moves from inside cell to outside
                   cell (and possibly outside of the grid)
                   --Decrement counter 2 for old cell
                   --Turn off inside_cell flag

           Case 3: Particle moves from inside one cell to inside
                   another cell
                   --Decrement counter 2 for old cell
                   --Increment counters 1-3 for new cell
                   --Store the coordinates of the particle's new cell
                     location

           Case 4: Particle moves and remains inside original
                   cell (does not leave cell)
                   --Increment counter 1

           Case 5: Particle moves and remains outside of cell
                   --No action.                                       */

      // Check if particle is still in computation grid
      if ((iX < grid_size) && (iY < grid_size) && (iX >= 0) && (iY >= 0)) {
        // Compare the radius to particle's distance from center of cell
        if (radius >= sycl::sqrt(dX * dX + dY * dY)) {
          // Satisfies counter 1 requirement for cases 1, 3, 4
          increment_C1[p] = true;
          // Case 1
          if (!inside_cell[p]) {
            increment_C2[p] = true;
            increment_C3[p] = true;
            inside_cell[p] = true;
            update_coordinates[p] = true;
          }
          // Case 3
          else if (prev_known_cell_coordinate_X[p] != iX ||
                   prev_known_cell_coordinate_Y[p] != iY) {
            increment_C2[p] = true;
            increment_C3[p] = true;
            update_coordinates[p] = true;
            decrement_C2_for_previous_cell[p] = true;
          }
          // Else: Case 4 --No action required. Counter 1 already updated

        }  // End inside cell if statement

        // Case 2a --Particle remained inside grid and moved outside cell
        else if (inside_cell[p]) {
          inside_cell[p] = false;
          decrement_C2_for_previous_cell[p] = true;
        }
        // Else: Case 5a --Particle remained inside grid and outside cell
        // --No action required

      }  // End inside grid if statement

      // Case 2b --Particle moved outside grid and outside cell
      else if (inside_cell[p]) {
        inside_cell[p] = false;
        decrement_C2_for_previous_cell[p] = true;
      }
      // Else: Case 5b --Particle remained outside of grid
      // --No action required

      // Index variable for 3rd dimension of grid
      size_t layer;
      // Current and previous cell coordinates
      size_t curr_coordinates = iX + iY * grid_size;
      size_t prev_coordinates = prev_known_cell_coordinate_X[p] +
                                prev_known_cell_coordinate_Y[p] * grid_size;
      // gs2 variable (used below) equals grid_size * grid_size
      //

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = gs2;
      if (decrement_C2_for_previous_cell[p]) --(grid[prev_coordinates + layer]);

      if (update_coordinates[p]) {
        prev_known_cell_coordinate_X[p] = iX;
        prev_known_cell_coordinate_Y[p] = iY;
      }

      // Counter 1 layer of the grid (0 * grid_size * grid_size)
      layer = 0;
      if (increment_C1[p]) ++(grid[curr_coordinates + layer]);

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = gs2;
      if (increment_C2[p]) ++(grid[curr_coordinates + layer]);

      // Counter 3 layer of the grid (2 * grid_size * grid_size)
      layer = gs2 + gs2;
      if (increment_C3[p]) ++(grid[curr_coordinates + layer]);

      increment_C1[p] = false;
      increment_C2[p] = false;
      increment_C3[p] = false;
      decrement_C2_for_previous_cell[p] = false;
      update_coordinates[p] = false;

    }  // Next iteration inner for loop
  }  // Next iteration outer for loop
  delete[] inside_cell;
  delete[] increment_C1;
  delete[] increment_C2;
  delete[] increment_C3;
  delete[] decrement_C2_for_previous_cell;
  delete[] update_coordinates;
  delete[] prev_known_cell_coordinate_X;
  delete[] prev_known_cell_coordinate_Y;
}  // End of function CPUParticleMotion()

// This function distributes simulation work and prints 10 snapshots
void CPUParticleMotionWithSnapshotPrinting(const int seed, float* particle_X, float* particle_Y,
                       float* random_X, float* random_Y, size_t* grid,
                       const size_t grid_size, const size_t planes,
                       const size_t n_particles, const size_t n_iterations,
                       const float radius, const unsigned int output_snapshot_flag) {
  // If user turned on snapshot printing, then print a snapshot every
  // average_iteration_to_print_snapshot snapshots
  // This code prints 10 snapshots per simulation, equally spaced apart in terms of the snapshot
  // (i.e., iteration) number. If n_iterations < 10, then this code prints n_iterations snapshots
  float res = 1.0f;
  if (n_iterations >= 10) res = (float)n_iterations / 10.0f;
  const float avg_iteration_to_print_snapshots = res;
  float iteration_to_print_snapshot = avg_iteration_to_print_snapshots;
  // Grid size squared
  const size_t gs2 = grid_size * grid_size;
  // Counter 2's layer (the instantaneous snapshot)
  size_t snapshot_layer = gs2;

  cout << "Running on: CPU\n";
  cout << "Number of iterations: " << n_iterations << "\n";
  cout << "Number of particles: " << n_particles << "\n";
  cout << "Size of the grid: " << grid_size << "\n";
  cout << "Random number seed: " << seed << "\n";

  // Array of flags for each particle.
  // Each array initialized to false/zero with new array zero initializer.
  // True when particle is found to be in a cell
  bool* inside_cell = new bool[n_particles]();
  // Operations flags
  bool* increment_C1 = new bool[n_particles]();
  bool* increment_C2 = new bool[n_particles]();
  bool* increment_C3 = new bool[n_particles]();
  bool* decrement_C2_for_previous_cell = new bool[n_particles]();
  bool* update_coordinates = new bool[n_particles]();
  // Coordinates of the last known cell this particle resided in.
  // Initialized in motion simulation algorithm below and presumably not used
  // before initialization. Thus, avoid spending computation time initializing
  unsigned int* prev_known_cell_coordinate_X = new unsigned int[n_particles];
  unsigned int* prev_known_cell_coordinate_Y = new unsigned int[n_particles];

  // Motion simulation algorithm
  // --Start iterations--
  // Each iteration:
  //    1. Updates the position of all particles
  //    2. Checks if particle is inside a cell or not
  //    3. Updates counters in cells array (grid)
  //

  // All n_particles particles need to each be displaced once per iteration
  // to match device's algorithm
  for (size_t iter = 0; iter < n_iterations; ++iter) {
    for (size_t p = 0; p < n_particles; ++p) {
      // Set the displacements to the random numbers
      float displacement_X = random_X[iter * n_particles + p];
      float displacement_Y = random_Y[iter * n_particles + p];
      // Displace particles
      particle_X[p] += displacement_X;
      particle_Y[p] += displacement_Y;
      // Compute distances from particle position to grid point i.e.,
      // the particle's distance from center of cell. Subtract the
      // integer value from floating point value to get just the
      // decimal portion. Use this value to later determine if the
      // particle is inside or outside of the cell
      float dX = sycl::abs(particle_X[p] - sycl::round(particle_X[p]));
      float dY = sycl::abs(particle_Y[p] - sycl::round(particle_Y[p]));
      /* Grid point indices closest the particle, defined by the following:
      ------------------------------------------------------------------
      |               Condition               |         Result         |
      |---------------------------------------|------------------------|
      |particle_X + 0.5 >= ceiling(particle_X)|iX = ceiling(particle_X)|
      |---------------------------------------|------------------------|
      |particle_Y + 0.5 >= ceiling(particle_Y)|iY = ceiling(particle_Y)|
      |---------------------------------------|------------------------|
      |particle_X + 0.5 < ceiling(particle_X) |iX = floor(particle_X)  |
      |---------------------------------------|------------------------|
      |particle_Y + 0.5 < ceiling(particle_Y) |iY = floor(particle_Y)  |
      ------------------------------------------------------------------  */
      int iX = sycl::floor(particle_X[p] + 0.5);
      int iY = sycl::floor(particle_Y[p] + 0.5);

      /* There are 5 cases when considering particle movement about the
         grid.

         All 5 cases are distinct from one another; i.e., any particle's
         motion falls under one and only one of the following cases:

           Case 1: Particle moves from outside cell to inside cell
                   --Increment counters 1-3
                   --Turn on inside_cell flag
                   --Store the coordinates of the
                     particle's new cell location

           Case 2: Particle moves from inside cell to outside
                   cell (and possibly outside of the grid)
                   --Decrement counter 2 for old cell
                   --Turn off inside_cell flag

           Case 3: Particle moves from inside one cell to inside
                   another cell
                   --Decrement counter 2 for old cell
                   --Increment counters 1-3 for new cell
                   --Store the coordinates of the particle's new cell
                     location

           Case 4: Particle moves and remains inside original
                   cell (does not leave cell)
                   --Increment counter 1

           Case 5: Particle moves and remains outside of cell
                   --No action.                                       */

      // Check if particle is still in computation grid
      if ((iX < grid_size) && (iY < grid_size) && (iX >= 0) && (iY >= 0)) {
        // Compare the radius to particle's distance from center of cell
        if (radius >= sycl::sqrt(dX * dX + dY * dY)) {
          // Satisfies counter 1 requirement for cases 1, 3, 4
          increment_C1[p] = true;
          // Case 1
          if (!inside_cell[p]) {
            increment_C2[p] = true;
            increment_C3[p] = true;
            inside_cell[p] = true;
            update_coordinates[p] = true;
          }
          // Case 3
          else if (prev_known_cell_coordinate_X[p] != iX ||
                   prev_known_cell_coordinate_Y[p] != iY) {
            increment_C2[p] = true;
            increment_C3[p] = true;
            update_coordinates[p] = true;
            decrement_C2_for_previous_cell[p] = true;
          }
          // Else: Case 4 --No action required. Counter 1 already updated

        }  // End inside cell if statement

        // Case 2a --Particle remained inside grid and moved outside cell
        else if (inside_cell[p]) {
          inside_cell[p] = false;
          decrement_C2_for_previous_cell[p] = true;
        }
        // Else: Case 5a --Particle remained inside grid and outside cell
        // --No action required

      }  // End inside grid if statement

      // Case 2b --Particle moved outside grid and outside cell
      else if (inside_cell[p]) {
        inside_cell[p] = false;
        decrement_C2_for_previous_cell[p] = true;
      }
      // Else: Case 5b --Particle remained outside of grid
      // --No action required

      // Index variable for 3rd dimension of grid
      size_t layer;
      // Current and previous cell coordinates
      size_t curr_coordinates = iX + iY * grid_size;
      size_t prev_coordinates = prev_known_cell_coordinate_X[p] +
                                prev_known_cell_coordinate_Y[p] * grid_size;
      // gs2 variable (used below) equals grid_size * grid_size
      //

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = gs2;
      if (decrement_C2_for_previous_cell[p]) --(grid[prev_coordinates + layer]);

      if (update_coordinates[p]) {
        prev_known_cell_coordinate_X[p] = iX;
        prev_known_cell_coordinate_Y[p] = iY;
      }

      // Counter 1 layer of the grid (0 * grid_size * grid_size)
      layer = 0;
      if (increment_C1[p]) ++(grid[curr_coordinates + layer]);

      // Counter 2 layer of the grid (1 * grid_size * grid_size)
      layer = gs2;
      if (increment_C2[p]) ++(grid[curr_coordinates + layer]);

      // Counter 3 layer of the grid (2 * grid_size * grid_size)
      layer = gs2 + gs2;
      if (increment_C3[p]) ++(grid[curr_coordinates + layer]);

      increment_C1[p] = false;
      increment_C2[p] = false;
      increment_C3[p] = false;
      decrement_C2_for_previous_cell[p] = false;
      update_coordinates[p] = false;

    }  // Next iteration inner for loop
    // Print different states of the grid mid-computation after updating
    // counters and moving all the particles, print this iteration's snapshot
    if (iter + 1 >= iteration_to_print_snapshot) {
      ClearScreen(3);
      PrintVectorAsMatrix<size_t>(&grid[snapshot_layer], grid_size,
                                  grid_size);
      // Calculate and print the total number of particles in the snapshot
      unsigned int psum = 0;
      for (unsigned int i = snapshot_layer; i < snapshot_layer + gs2; ++i) psum += grid[i];
      cout << "Number of particles inside snapshot: " << psum << "\n";
      unsigned int retv = sleep(1);
      // err check
      iteration_to_print_snapshot += avg_iteration_to_print_snapshots;
    }
  }  // Next iteration outer for loop
  delete[] inside_cell;
  delete[] increment_C1;
  delete[] increment_C2;
  delete[] increment_C3;
  delete[] decrement_C2_for_previous_cell;
  delete[] update_coordinates;
  delete[] prev_known_cell_coordinate_X;
  delete[] prev_known_cell_coordinate_Y;
}  // End of function CPUParticleMotionWithSnapshotPrinting()
