CXX := clang++
PROTOC := protoc
CXXFLAGS := -std=c++14 -Wall -O2 -pthread
LDFLAGS := -L/usr/local/lib `pkg-config --cflags --libs protobuf`
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
PROTO_HEADERS := $(PROTOS:%.proto=%.pb.h)
CNN_OBJS := $(CNN_SRCS:%.cc=$(OBJDIR)/%.o)
GA_OBJS := $(GA_SRCS:%.cc=$(OBJDIR)/%.o)
CNN_DEPS := $(CNN_SRCS:%.cc=%.d)
GA_DEPS := $(GA_SRCS:%.cc=%.d)

.PHONY: all
all: cnn ga utest

.PHONY: cnn
cnn: $(BINDIR)/cnn

.PHONY: ga
ga: $(BINDIR)/ga

.PHONY: utest
utest: $(BINDIR)/utest

$(BINDIR)/cnn: $(CNN_OBJS) $(BINDIR)
	$(CXX) -o $@ $(CNN_OBJS) $(LDFLAGS)

$(BINDIR)/ga: $(GA_OBJS) $(BINDIR)
	$(CXX) -o $@ $(GA_OBJS) $(LDFLAGS)

$(BINDIR)/utest: $(TEST_SRCS) $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRCS) -I$(GTEST_INCLUDEDIR) $(INCLUDES) -L$(GTEST_LIBDIR) $(TESTFLAGS)

$(OBJDIR)/%.o: %.cc $(PROTO_HEADERS)
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ -MMD $<

.SECONDARY:
%.pb.cc %.pb.h: %.proto
	$(PROTOC) -I=$(PROTODIR) --cpp_out=$(PROTODIR) $<

$(BINDIR):
	mkdir -p $(BINDIR)

.PHONY: clean
clean:
	rm -rf $(BINDIR) $(OBJDIR) $(PROTODIR)/*.pb.*

-include $(CNN_DEPS) $(GA_DEPS)
