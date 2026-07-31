include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_gelqt3

example_gelqt3: example_gelqt3.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_gelqt3