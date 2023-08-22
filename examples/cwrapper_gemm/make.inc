tlapack_inc = /usr/local/include
tlapack_lib = /usr/local/lib

CFLAGS = -I$(tlapack_inc)
LD = g++
LDFLAGS  = -L$(tlapack_lib) -ltlapack_c -Wl,-rpath=$(tlapack_lib)
