INSTTOOL=/usr/bin/install
INSTALL=/usr/local

install:
	$(INSTTOOL) -d $(INSTALL)/bin
	$(INSTTOOL) -d $(INSTALL)/doc/doxygen
	$(INSTTOOL) -d $(INSTALL)/man/man1
	$(INSTTOOL) -m 755 bin/doxygen    $(INSTALL)/bin
	cp -r html $(INSTALL)/doc/doxygen
	cp -r examples $(INSTALL)/doc/doxygen
	cp doxygen_manual*.pdf $(INSTALL)/doc/doxygen
	cp man/man1/* $(INSTALL)/man/man1/
