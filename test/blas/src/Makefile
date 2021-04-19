ROOTDIR = ..
-include $(ROOTDIR)/make.inc

CXXFLAGS += -I$(ROOTDIR)

#-------------------------------------------------------------------------------
# Rules
.PHONY: all
all: $(patsubst %.cpp,%.o,$(wildcard *.cpp))

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o