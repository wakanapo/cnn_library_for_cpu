CXX := clang++
PROTOC := protoc
CXXFLAGS := -std=c++14 -Wall -O2 -pthread
LDFLAGS := -L/usr/local/lib
TESTFLAGS := -lgtest_main -lgtest -lpthread

GTESTDIR := $(shell echo "$(HOME)")/googletest-master
GTEST_INCLUDEDIR := $(GTESTDIR)/googletest/include
GTEST_LIBS := $(GTESTDIR)/build/googlemock/gtest
INCLUDES := -I./src -I./include
BINDIR := bin
OBJDIR := obj
TESTDIR := test
PROTODIR := src/protos

PROTOS := $(wildcard src/protos/*.proto)
UTILS := $(wildcard src/util/*.cc)
CNN_SRCS := $(wildcard src/cnn/*.cc) $(UTILS) $(PROTOS:%.proto=%.pb.cc) 
GA_SRCS := $(wildcard src/ga/*.cc) $(UTILS) $(PROTOS:%.proto=%.pb.cc)
TEST_SRCS := test/util_test.cc src/util/converter.cc src/protos/arithmatic.pb.cc  src/util/flags.cc
CNN_OBJS := $(CNN_SRCS:%.cc=$(OBJDIR)/%.o)
GA_OBJS := $(GA_SRCS:%.cc=$(OBJDIR)/%.o)
CNN_DEPS := $(CNN_SRCS:%.cc=%.d)
GA_DEPS := $(GA_SRCS:%.cc=%.d)

.PHONY: all
all: protoc cnn ga utest

protoc: $(PROTOS)
	$(PROTOC) -I=$(PROTODIR) --cpp_out=$(PROTODIR) $^

$(OBJDIR)/%.o: %.cc $(BINDIR)
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ -MMD $<

cnn: $(CNN_OBJS)
	$(CXX) -o $(BINDIR)/$@ $^ $(LDFLAGS) `pkg-config --cflags --libs protobuf`

ga: $(GA_OBJS)
	$(CXX) -o $(BINDIR)/$@ $^ $(LDFLAGS) `pkg-config --cflags --libs protobuf`

utest: $(TEST_SRCS) $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $(BINDIR)/$@ $(TEST_SRCS) -I$(GTEST_INCLUDEDIR) $(INCLUDES) -L$(GTEST_LIBDIR) $(TESTFLAGS) `pkg-config --cflags --libs protobuf`

-include $(CNN_DEPS) $(GA_DEPS)

.PHONY: clean
clean:
	rm -rf $(BINDIR) $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

