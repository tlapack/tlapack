-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_create_float_library

# Create library tlapack_EigenMatrixXf.a
tlapack_EigenMatrixXf.a: tlapack_EigenMatrixXf.o
	$(AR) $(ARFLAGS) $@ $<

example_create_float_library: main.o tlapack_EigenMatrixXf.a
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS) tlapack_EigenMatrixXf.a

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_create_float_library tlapack_EigenMatrixXf.a