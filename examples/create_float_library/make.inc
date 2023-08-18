#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc = ../../include
eigen_inc   = /usr/include/eigen3

CXXFLAGS = -g -I$(tlapack_inc) -I$(eigen_inc) -Wall -pedantic
LDFLAGS  = 