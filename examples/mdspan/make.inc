#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc = /usr/local/include
tlapack_lib = /usr/local/lib
mdspan_inc  = /usr/local/include

CXXFLAGS = -I$(tlapack_inc) -I$(mdspan_inc) -Wall -pedantic
LDFLAGS  = -L$(tlapack_lib)
