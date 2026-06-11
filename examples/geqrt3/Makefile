include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_geqrt3

example_geqrt3: example_geqrt3.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_geqrt3
	