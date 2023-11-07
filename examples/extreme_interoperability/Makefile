-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_interoperability

example_interoperability : example_interoperability.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_interoperability