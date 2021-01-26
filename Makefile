CXX := clang++
PROTOC := protoc
ENABLE_LOGGING := 0
CXXFLAGS := -std=c++14 -g3 -Wall -Wextra -O2 -pthread -DENABLE_LOGGING=$(ENABLE_LOGGING)
LDFLAGS := -L/usr/local/lib -lgrpc++ `pkg-config --cflags --libs protobuf`
TESTFLAGS := -lgtest_main -lgtest -lpthread -lgrpc++ `pkg-config --cflags --libs protobuf`

INCLUDES := -I./src -I./include
BINDIR := bin
OBJDIR := obj
TESTDIR := test
PROTODIR := src/protos

include src/include.mk

UTILS := $(wildcard src/util/*.cc)
CNN_SRCS := $(wildcard src/cnn/*.cc) $(UTILS) $(PROTO_MESSAGES:%.proto=%.pb.cc) $(PROTO_SERVICES:%.proto=%.grpc.pb.cc)
GA_SRCS := $(wildcard src/ga/*.cc) $(UTILS) $(PROTO_MESSAGES:%.proto=%.pb.cc) $(PROTO_SERVICES:%.proto=%.grpc.pb.cc)
TEST_SRCS := test/util_test.cc src/util/converter.cc src/protos/arithmatic.pb.cc  src/util/flags.cc
PROTO_HEADERS := $(PROTO_MESSAGES:%.proto=%.pb.h) $(PROTO_SERVICES:%.proto=%.grpc.pb.h)
CNN_OBJS := $(CNN_SRCS:%.cc=$(OBJDIR)/%.o)
GA_OBJS := $(GA_SRCS:%.cc=$(OBJDIR)/%.o)
CNN_DEPS := $(CNN_SRCS:%.cc=$(OBJDIR)/%.d)
GA_DEPS := $(GA_SRCS:%.cc=$(OBJDIR)/%.d)
TEST_DEPS := $(OBJDIR)/test/util_test.d

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
	@if [ ! -e  `dirname $(TEST_DEPS)` ]; then mkdir -p `dirname $(TEST_DEPS)`; fi
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TESTFLAGS) -o $@ $(TEST_SRCS) -MMD -MF $(TEST_DEPS)

$(OBJDIR)/%.o: %.cc $(PROTO_HEADERS)
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ -MMD $<

.SECONDARY:
%.grpc.pb.cc %.grpc.pb.h: %.proto
	$(PROTOC) -I=$(PROTODIR) --grpc_out=$(PROTODIR) --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` $<

.SECONDARY:
%.pb.cc %.pb.h: %.proto
	$(PROTOC) -I=$(PROTODIR) --cpp_out=$(PROTODIR) $<

$(BINDIR):
	mkdir -p $(BINDIR)

.PHONY: clean
clean:
	rm -rf $(BINDIR) $(OBJDIR) $(PROTODIR)/*.pb.*

-include $(CNN_DEPS) $(GA_DEPS) $(TEST_DEPS)
