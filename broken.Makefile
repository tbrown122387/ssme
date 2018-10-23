
TODO: this doesn't work!



# Specify extensions of files to delete when cleaning
CLEANEXTS := o a 

# Specify the target file and the install directory
OUTPUTFILE := libssme.a
INSTALLDIR := ./binaries


# Default target
.PHONY: all
all: $(OUTPUTFILE)

# Build libssme.a from the .o files
$(OUTPUTFILE): param_pack.o param_transforms.o
	ar ru $@ $^
	ranlib $@

# No rule to build .o object files from .cpp  
# files is required; this is handled by make's database of
# implicit rules
CXXFLAGS := -I/usr/include/eigen3 -I./include -I~/pf/include

.PHONY: install
install:
	mkdir -p $(INSTALLDIR)
	cp -p $(OUTPUTFILE) $(INSTALLDIR)

.PHONY: clean 
clean:
	for file in $(CLEANEXTS); do rm -f *.$$file; done

# Indicate dependencies of .ccp files on .hpp files
param_pack.o: param_pack.h param_transforms.h
param_transforms.o: param_transforms.h
