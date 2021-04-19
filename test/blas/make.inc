# Configure this:
# ---------------
catch_inc = /home/weslley/repositories/ballistic/tlapack/thirdparty/
blaspp_inc = /home/weslley/repositories/blaspp/include/
blaspp_lib = /home/weslley/repositories/blaspp/lib/
blas_lib = /home/weslley/repositories/lapack/build_forblaspp/lib/
# ---------------

CXX = g++
CXXFLAGS = -I$(blaspp_inc) -I$(catch_inc) -g3 -Wall -Wextra
# CXXFLAGS = -I$(blaspp_inc) -g3

LDFLAGS = -L$(blaspp_lib) -L$(blas_lib) -lblaspp -lblas -lgfortran

AR = ar
ARFLAGS = cr
RANLIB = ranlib