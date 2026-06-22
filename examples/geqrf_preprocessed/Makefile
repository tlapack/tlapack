-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_geqrf_preprocessed

example_geqrf_preprocessed: example_geqrf_preprocessed.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_geqrf_preprocessed
	