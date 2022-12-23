-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: performance_eigen performance_tlapack

performance_eigen: performance_eigen.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
performance_tlapack: performance_tlapack.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f performance_eigen
	rm -f performance_tlapack