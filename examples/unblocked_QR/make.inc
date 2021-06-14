#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc = 
tlapack_lib = 

CXXFLAGS = -I$(tlapack_inc) -Wall -pedantic
LDFLAGS  = -L$(tlapack_lib) -ltblas