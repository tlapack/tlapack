tlapack_inc = ../../include
eigen_inc   = /usr/include/local/eigen3

CXXFLAGS = -I$(tlapack_inc) -I$(eigen_inc)
LDFLAGS  = 