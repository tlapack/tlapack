-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_genQR

example_genQR: example_genQR.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_genQR