ifndef ROOTDIR
	ROOTDIR = .
endif
-include $(ROOTDIR)/make.inc

#-------------------------------------------------------------------------------
# Rules
.PHONY: all
all: tester run-tests
FORCE: 

clean:
	$(MAKE) -C src $@
	rm -f tester tests_main.o

#-------------------------------------------------------------------------------
# Build corner cases
corner-cases:
	(cd scripts && ./wrapper_tests.py > ../src/test_corner_cases.cpp)

#-------------------------------------------------------------------------------
# Test main file
test_obj = $(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

tests_main.o: tests_main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src: FORCE
	$(MAKE) -C $@

tester: tests_main.o src
	$(CXX) $(CXXFLAGS) -o $@ $< $(test_obj) $(LDFLAGS)

run-tests: tester
	./tester