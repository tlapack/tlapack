#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc = /usr/local/include
tlapack_lib = /usr/local/lib

CXXFLAGS = -I$(tlapack_inc) -Wall -pedantic # -DUSE_MPFR
LDFLAGS  = -L$(tlapack_lib) # -lmpfr -lgmp
