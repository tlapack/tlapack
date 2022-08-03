#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc     = /usr/local/include
tlapack_lib     = /usr/local/lib

FFLAGS = -I$(tlapack_inc) -Wall -pedantic
LDFLAGS  = -L$(tlapack_lib) -ltlapack_fortran -ltlapack_c -lgfortran

LD = g++
