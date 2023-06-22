#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc = /usr/local/include
tlapack_lib = /usr/local/lib

CXXFLAGS = -I$(tlapack_inc) -Wall -pedantic -std=c++17 -D USE_LAPACK
FFLAGS = 
LDFLAGS  = -L$(tlapack_lib) -llapack -lblas -lm -lgfortran