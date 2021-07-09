-include make.inc

#-------------------------------------------------------------------------------
# Executables

all: example_cwrapper_gemm

example_cwrapper_gemm: example_cwrapper_gemm.o
	$(LD) $(CFLAGS) -o $@ $< $(LDFLAGS)

#-------------------------------------------------------------------------------
# Rules

.PHONY: all

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o
	rm -f example_cwrapper_gemm