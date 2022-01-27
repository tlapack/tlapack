#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc = /usr/local/include
tlapack_lib = /usr/local/lib
eigen_inc   = 
mdspan_inc  = 

CXXFLAGS = -g -I$(tlapack_inc) -I$(eigen_inc) -I$(mdspan_inc) -Wall -pedantic
LDFLAGS  = 