-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_geqr2

example_geqr2: example_geqr2.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_geqr2