tlapack_mod = /usr/local/include/tlapack/fortran
tlapack_lib = /usr/local/lib

FFLAGS = -I$(tlapack_mod)
LD = g++
LDFLAGS  = -L$(tlapack_lib) -ltlapack_fortran -ltlapack_c -lgfortran -Wl,-rpath=$(tlapack_lib) -lgfortran
