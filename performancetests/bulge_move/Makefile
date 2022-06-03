CXXFLAGS=-g -O3 -fPIC -Wall -std=c++17 -DNDEBUG -DUSE_BLASPP_WRAPPERS
LFLAGS=-L/opt/slate/lib -I/opt/slate/include -lstdc++ -lgfortran -lblaspp -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl


all: profile_recursive_bulge_move

profile_recursive_bulge_move : profile_recursive_bulge_move.cpp
	g++ $(CXXFLAGS) -o $@ $< -I../../include $(LFLAGS)

# Some phony commands
.PHONY: clean

clean:
	rm -f ./*.mod ./*.o ./profile_recursive_bulge_move