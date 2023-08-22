-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_fortranModule_caxpy

example_fortranModule_caxpy: example_fortranModule_caxpy.o
	$(LD) $(FFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

example_fortranModule_caxpy.o: example_fortranModule_caxpy.f90
	$(FC) $(FFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_fortranModule_caxpy
	