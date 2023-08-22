tlapack_inc = /usr/local/include
eigen_inc   = /usr/local/include/eigen3

CXXFLAGS = -I$(tlapack_inc) -I$(eigen_inc)
FFLAGS = 
LDFLAGS  = -llapack -lblas -lm -lgfortran