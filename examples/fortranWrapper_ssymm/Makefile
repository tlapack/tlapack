-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_fortranWrapper_ssymm

example_fortranWrapper_ssymm: example_fortranWrapper_ssymm.o
	$(LD) $(FFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

%.o: %.f90 $(tlapack_inc)/constants.mod
	$(FC) $(FFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_fortranWrapper_ssymm
	
.o.mod:
	@true
constants.o: $(tlapack_inc)/blas/constants.f90
	$(FC) $(FFLAGS) -c -o $@ $<
