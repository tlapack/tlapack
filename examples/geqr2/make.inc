#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc = /opt/tlapack/include #/usr/local/include
tlapack_lib = /opt/tlapack/lib #/usr/local/lib

CXXFLAGS = -I$(tlapack_inc) -Wall -pedantic
LDFLAGS  = -L$(tlapack_lib)
