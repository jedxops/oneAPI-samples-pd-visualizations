# *******************************************************************
# *                            Description                          *
# *******************************************************************

# Test particle diffusion

# *******************************************************************
# *                              Usage                              *
# *******************************************************************

# python3 regression_test.py

# *******************************************************************
# *                             Run commands                        *
# *******************************************************************

import os
import argparse

testTypes=["particles", "powers_of_2_particles", "particles0260", "iterations",
           "grid_size", "all"]
deviceTypes=["cpu", "CPU", "gpu", "GPU", "acc", "ACC", '']
beTypes=["PI_LEVEL0", "PI_LEVEL_ZERO", "PI_OPENCL", "pi_level0", "pi_level_zero", "pi_opencl"]

# Parse CLAs
prser = argparse.ArgumentParser(description="Parse flags")

prser.add_argument("-c", "--cpu", help="Specify CPU flag for" +
                   " program execution.", type=str, default="0")
prser.add_argument("-o", "--output", help="Specify output flag" +
                   " for program execution.", type=str, default="0")
prser.add_argument("-r", "--rngSeed", help="Specify RNG seed for"
                   " program execution.", type=str, default="777")
prser.add_argument("-s", "--snapshot", help="Snapshot output on/off",
                   type=str, default="0")
prser.add_argument("-t", "--testType", help="Specify type of test" +
                   " to be conducted for program execution.",
                   type=str, choices=testTypes, default="all")
prser.add_argument("-d", "--device", help="Specify type of sycl device to use",
                   type=str, choices=deviceTypes, default='')

prser.add_argument("-b", "--backend", help="Specify type of sycl backend to use",
                   type=str, choices=beTypes, default='')

arguments = prser.parse_args()
cpuFlg = arguments.cpu
outputFlg = arguments.output
randSeed = arguments.rngSeed
testType = arguments.testType
snapshot = arguments.snapshot

backend_device_string = ''
if arguments.backend != '':
  backend = "SYCL_BE=" + (arguments.backend.upper()) + " "
  backend_device_string += backend

if arguments.device != '':
  device = "SYCL_DEVICE_TYPE=" + arguments.device.upper() + " "
  backend_device_string += "&& " + device + " "
  # Turn CPU flag off if the user requested a non-CPU device,
  # so the CPU particle motion function doesn't try to use a GPU device (avoids a seg fault)
  if device != "SYCL_DEVICE_TYPE=CPU ": cpuFlg = '0'

claFlags = " " + randSeed + " " + cpuFlg + " " + snapshot + " " + outputFlg

ptest = 0
p2ptest = 0
p0260test = 0
itest = 0
gstest = 0

if testType == testTypes[0]: ptest = 1
elif testType == testTypes[1]: p2ptest = 1
elif testType == testTypes[2]: p0260test = 1
elif testType == testTypes[3]: itest = 1
elif testType == testTypes[4]: gstest = 1
elif testType == testTypes[5]:
  ptest = 1
  p2ptest = 1
  p0260test = 1
  itest = 1
  gstest = 1

# Compile
print("Compiling ...\n")
print("dpcpp -fsycl -std=c++17 -O3 main.cpp utils.cpp motionsim_kernel.cpp motionsim.cpp -o pd -lmkl_sycl -lmkl_intel_ilp64"+
          " -lmkl_sequential -lmkl_core\n")

os.system("dpcpp -fsycl -std=c++17 -O3 main.cpp utils.cpp motionsim_kernel.cpp motionsim.cpp -o pd -lmkl_sycl -lmkl_intel_ilp64"+
          " -lmkl_sequential -lmkl_core")

# *******************************************************************
# *                     Scale Arguments and Execute Code            *
# *******************************************************************

if ptest == 1:
  print("********************VARYING PARTICLE COUNTS********************")
  print(backend_device_string + "./pd 10000 1 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 1 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 5 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 5 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 10 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 10 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 20 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 20 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 25 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 25 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 40 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 40 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 100 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 100 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 150 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 150 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 200 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 200 21" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 256 21" + claFlags)
  os.system(backend_device_string + "./pd 10000 256 21" + claFlags)
  print("-----------------------------------------------------\n\n")




  # takes a very long time to finish running (> 1 hr):
  # print("\n" + backend_device_string + "./pd 10000 50 21" + claFlags)
  # os.system(backend_device_string + "./pd 10000 50 21" + claFlags)
  # print("-----------------------------------------------------\n\n")

if p2ptest == 1:
  print("************VARYING PARTICLE COUNTS BY POWERS OF 2************")
  i = 2
  while i <= 65536:
    print("\n" + backend_device_string + "./pd 100 " + str(i) + " 21" + claFlags)
    os.system(backend_device_string + "./pd 100 " + str(i) + " 21" + claFlags)
    print("-----------------------------------------------------\n\n")
    i = i * 2

  i = 256
  j = 0
  while j < 20:
    print("\n" + backend_device_string + "./pd 1000  " + str(i) + " 21" + claFlags)
    os.system(backend_device_string + "./pd 1000 " + str(i) + " 21" + claFlags)
    print("-----------------------------------------------------\n\n")
    j = j + 1

if p0260test == 1:
  i = 1
  while i <= 260:
    print("\n" + backend_device_string + "./pd 100 " + str(i) + " 21" + claFlags)
    os.system(backend_device_string + "./pd 100 " + str(i) + " 21" + claFlags)
    print("-----------------------------------------------------\n\n")
    i = i + 1

if gstest == 1:
  print("********************VARYING GRID SIZE********************")
  print("\n" + backend_device_string + "./pd 10000 25 1" + claFlags)
  os.system(backend_device_string + "./pd 10000 25 1" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 25 5" + claFlags)
  os.system(backend_device_string + "./pd 10000 25 5" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 25 10" + claFlags)
  os.system(backend_device_string + "./pd 10000 25 10" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 25 15" + claFlags)
  os.system(backend_device_string + "./pd 10000 25 15" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 25 20" + claFlags)
  os.system(backend_device_string + "./pd 10000 25 20" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 25 30" + claFlags)
  os.system(backend_device_string + "./pd 10000 25 30" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10000 25 40" + claFlags)
  os.system(backend_device_string + "./pd 10000 25 40" + claFlags)
  print("-----------------------------------------------------\n\n")
  # takes a very long time to finish running (> 1 hr):
  # print("\n" + backend_device_string + "./pd 10000 25 50" + claFlags)
  # os.system(backend_device_string + "./pd 10000 25 50" + claFlags)
  # print("-----------------------------------------------------\n\n")

if itest == 1:
  print("********************VARYING NUMBER OF ITERATIONS********************")
  print("\n" + backend_device_string + "./pd 1 25 50" + claFlags)
  os.system(backend_device_string + "./pd 1 25 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 10 25 50" + claFlags)
  os.system(backend_device_string + "./pd 10 25 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 100 25 50" + claFlags)
  os.system(backend_device_string + "./pd 100 25 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 1000 25 50" + claFlags)
  os.system(backend_device_string + "./pd 1000 25 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  print("\n" + backend_device_string + "./pd 5000 25 50" + claFlags)
  os.system(backend_device_string + "./pd 5000 25 50" + claFlags)
  print("-----------------------------------------------------\n\n")
  # takes a very long time to finish running (> 1 hr):
  # print("\n" + backend_device_string + "./pd 10000 25 50" + claFlags)
  # os.system(backend_device_string + "./pd 10000 25 50" + claFlags)
  # print("-----------------------------------------------------\n\n")
  # takes a very long time to finish running (> 1 hr):
  # print("\n" + backend_device_string + "./pd 100000 25 50" + claFlags)
  # os.system(backend_device_string + "./pd 100000 25 50" + claFlags)
  # print("-----------------------------------------------------\n\n")

print("Other tests:")
print("-----------------------------------------------------\n\n")
print("0 iteration test:")
print("\n" + backend_device_string + "./pd 0 25 50" + claFlags)
os.system(backend_device_string + "./pd 0 25 50" + claFlags)
print("-----------------------------------------------------\n\n")

os.system("rm pd ")

# *******************************************************************
# *                               Done                              *
# *******************************************************************
