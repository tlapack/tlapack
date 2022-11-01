CXXFLAGS=-g -O3 -fPIC -Wall -std=c++17 -DNDEBUG -DUSE_LAPACKPP_WRAPPERS
FFLAGS=-g -O3 -fPIC
LFLAGS=-L/home/weslleyp/storage/lapackpp/build_RelWithDebInfo/lib -L/home/weslleyp/storage/blaspp/build_RelWithDebInfo/lib -I/home/weslleyp/storage/lapackpp/build_RelWithDebInfo/include -I/home/weslleyp/storage/blaspp/build_RelWithDebInfo/include -lstdc++ -lgfortran -llapackpp -lblaspp /usr/lib/x86_64-linux-gnu/liblapack.so /usr/lib/x86_64-linux-gnu/libblas.so -lm -ldl

FINC=fortran_wrappers.o slahqr.o slaqr0.o slaqr1.o slaqr2.o slaqr3.o slaqr4.o slaqr5.o

all: profile_multishift_qr profile_aed

slahqr.o : slahqr.f
	gfortran $(FFLAGS) -c -o $@ $<

slaqr0.o : slaqr0.f
	gfortran $(FFLAGS) -c -o $@ $<

slaqr1.o : slaqr1.f
	gfortran $(FFLAGS) -c -o $@ $<

slaqr2.o : slaqr2.f
	gfortran $(FFLAGS) -c -o $@ $<

slaqr3.o : slaqr3.f
	gfortran $(FFLAGS) -c -o $@ $<

slaqr4.o : slaqr4.f
	gfortran $(FFLAGS) -c -o $@ $<

slaqr5.o : slaqr5.f
	gfortran $(FFLAGS) -c -o $@ $<

fortran_wrappers.o : fortran_wrappers.f90
	gfortran $(FFLAGS) -c -o $@ $<

profile_multishift_qr : profile_multishift_qr.cpp $(FINC)
	g++ $(CXXFLAGS) -o $@ $< $(FINC) -I../../include $(LFLAGS)

profile_aed : profile_aed.cpp $(FINC)
	g++ $(CXXFLAGS) -o $@ $< $(FINC) -I../../include $(LFLAGS)


# Some phony commands
.PHONY: clean

clean:
	rm -f ./*.mod ./*.o ./profile_multishift_qr ./profile_aed