#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_mod = /usr/local/include/tlapack/fortran
tlapack_lib = /usr/local/lib

FFLAGS = -I$(tlapack_mod) -Wall -pedantic
LDFLAGS  = -L$(tlapack_lib) -ltlapack_fortran -ltlapack_c -lgfortran

LD = g++
