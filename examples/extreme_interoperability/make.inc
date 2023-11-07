tlapack_inc = /usr/local/include
eigen_inc   = /usr/local/include/eigen3
mdspan_inc  = /usr/local/include

CXXFLAGS = -I$(tlapack_inc) -I$(eigen_inc) -I$(mdspan_inc)
LDFLAGS  = 
