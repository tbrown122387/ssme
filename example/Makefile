BIN := main
SRC_DIR := .
SRCS := $(notdir $(wildcard $(SRC_DIR)/*.cpp))
OBJDIR := .o
DEPDIR := .d

# object files, auto generated from source files
OBJS := $(patsubst %,$(OBJDIR)/%.o,$(basename $(SRCS)))

# compilers (at least gcc and clang) don't create the subdirectories automatically
$(shell mkdir -p $(DEPDIR))
$(shell mkdir -p $(dir $(OBJS)) >/dev/null)

# compiler and linker stuff
CXX := g++
LD := g++
CXXFLAGS := -std=c++11 -g -pg -Wall -O3 # use this one for speed
#CXXFLAGS := -std=c++11 -g -Wall -pg  # use this one for debugging
CPPFLAGS := -I/usr/include/eigen3 -I$$HOME/pf/include -I$$HOME/ssme/include
LDFLAGS := "-L$$HOME/pf/bin" "-L$$HOME/ssme/binaries"
DEPFLAGS = -MT $@ -MD -MP -MF $(DEPDIR)/$*.Td
LDLIBS := -lpf -lssme -lpthread

# compile C++ source files
COMPILE.cc = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(CPPFLAGS) -c -o $@
# link object files to binary
LINK.o = $(LD) $(LDFLAGS) -o $@ 
#LINK.o = $(LD) $(LDFLAGS) -pg -o $@ // use for gprof
# precompile step
PRECOMPILE =
# postcompile step
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

all: $(BIN)

.PHONY: clean
clean:
	$(RM) -r $(OBJDIR) $(DEPDIR)

.PHONY: help
help:
	@echo available targets: all clean

$(BIN): $(OBJS)
	$(LINK.o) $^ $(LDLIBS)

$(OBJDIR)/%.o: %.cpp
$(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d
	$(PRECOMPILE)
	$(COMPILE.cc) $<
	$(POSTCOMPILE)

.PRECIOUS = $(DEPDIR)/%.d
$(DEPDIR)/%.d: ;

-include $(DEPS)
