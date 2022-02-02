#-------------------------------------------------------------------------------
# <T>LAPACK library
tlapack_inc = /home/weslleyp/tlapack-test/include
tlapack_lib = /home/weslleyp/tlapack-test/lib
eigen_inc   = /home/weslleyp/storage/eigen
mdspan_inc  = 

CXXFLAGS = -g -I$(tlapack_inc) -I$(eigen_inc) -I$(mdspan_inc) -Wall -pedantic
LDFLAGS  = 