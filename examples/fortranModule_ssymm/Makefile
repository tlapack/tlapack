-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_fortranModule_ssymm

example_fortranModule_ssymm: example_fortranModule_ssymm.o
	$(LD) $(FFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

example_fortranModule_ssymm.o: example_fortranModule_ssymm.f90
	$(FC) $(FFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_fortranModule_ssymm
