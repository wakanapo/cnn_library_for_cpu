COMPILER = g++
CFLAGS = -Wall -O2
TESTFLAGS = -lgtest_main -lgtest -lpthread
CXXFLAGS = -std=c++11

BINDIR := bin

GTESTDIR = $(shell echo "$(HOME)")/googletest-master
GTEST_INCLUDEDIR = $(GTESTDIR)/googletest/include
GTEST_LIBS = $(GTESTDIR)/build/googlemock/gtest

INCLUDEDIR = $(shell pwd)/include

SRCDIR = $(shell pwd)/src
TESTDIR = $(shell pwd)/test

.PHONY: all
all: $(BINDIR)/mlp $(BINDIR)/cnn $(BINDIR)/utest

$(BINDIR)/mlp: src/mlp/mlp_main.cpp src/util/converter.cpp $(BINDIR)
	$(COMPILER) $(CXXFLAGS) -o $@ src/mlp/mlp_main.cpp src/protos/cnn_params.pb.cc src/util/converter.cpp -I$(SRCDIR) -I$(INCLUDEDIR) $(CFLAGS) `pkg-config --cflags --libs protobuf`

$(BINDIR)/cnn: src/cnn/cnn_main.cpp src/protos/cnn_params.pb.cc src/protos/arithmatic.pb.cc src/util/converter.cpp src/util/flags.cpp $(BINDIR)
	$(COMPILER) $(CXXFLAGS) -o $@ src/cnn/cnn_main.cpp src/protos/cnn_params.pb.cc src/protos/arithmatic.pb.cc src/util/converter.cpp src/util/flags.cpp -I$(SRCDIR) -I$(INCLUDEDIR) $(CFLAGS) `pkg-config --cflags --libs protobuf`

$(BINDIR)/protoc: src/protos/cnn_params.proto src/protos/arithmatic.proto
	protoc -I=src/protos --cpp_out=src/protos src/protos/cnn_params.proto src/protos/arithmatic.proto

$(BINDIR)/doubleTo: src/util/float_type.cpp src/protos/cnn_params.pb.cc $(BINDIR)
	$(COMPILER) $(CXXFLAGS) -o $@ src/util/float_type.cpp src/protos/cnn_params.pb.cc -I$(SRCDIR) -I$(INCLUDEDIR) $(CFLAGS) `pkg-config --cflags --libs protobuf`

$(BINDIR)/utest: test/util_test.cpp src/util/converter.cpp $(BINDIR)
	$(COMPILER) $(CXXFLAGS) -o $@ test/util_test.cpp src/util/converter.cpp -I$(GTEST_INCLUDEDIR) -I$(SRCDIR) -I$(INCLUDEDIR) -L$(GTEST_LIBDIR) $(CFLAGS) $(TESTFLAGS)

.PHONY: clean
clean:
	rm -rf $(BINDIR)/utest $(BINDIR)/mlp $(BINDIR)/cnn/float $(BINDIR)/cnn test/util_test.o src/mlp/mlp_main.o src/cnn/cnn_main.o src/cnn/float/cnn_main.o src/util/flags.o $(BINDIR)

$(BINDIR):
	mkdir -p $(BINDIR)
