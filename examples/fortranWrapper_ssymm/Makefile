-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_fortranWrapper_ssymm

example_fortranWrapper_ssymm: example_fortranWrapper_ssymm.o
	$(LD) $(FFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

example_fortranWrapper_ssymm.o: example_fortranWrapper_ssymm.f90
	$(FC) $(FFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_fortranWrapper_ssymm
