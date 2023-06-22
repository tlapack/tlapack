-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_eigenvalues profile_aed

ALLOBJ = src/slahqr.o src/slaqr0.o src/slaqr1.o src/slaqr2.o src/slaqr3.o src/slaqr4.o src/slaqr5.o src/fortran_wrappers.o

example_eigenvalues: example_eigenvalues.o $(ALLOBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

profile_aed: profile_aed.o $(ALLOBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.f.o:
	$(FC) $(FFLAGS) -c -o $@ $<
.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/fortran_wrappers.o: src/fortran_wrappers.f90
	$(FC) $(FFLAGS) -c -o $@ $<

clean:
	rm -f $(ALLOBJ) example_eigenvalues.o profile_aed.o
	rm -f example_eigenvalues profile_aed