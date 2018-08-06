#include <iostream>
#include <map>
#include <string>

#include "util/flags.hpp"

typedef std::map<std::string, void(*)(std::string)> flags_type;

namespace {
  int kExponent = 8;
  int kMantissa = 23;

  bool g_train = true;
  bool g_save_params = false;
  bool g_save_arithmetic = false;
  Type g_type = FLOAT;
  int g_expoent = kExponent;
  int g_mantissa = kMantissa;
  std::string g_weights_input;
  std::string g_weights_output;
  std::string g_arithmatic_output;
  float g_mutation_rate = 0.1;
  float g_cross_rate = 0.5;
  int g_max_generation = 30;
  std::string g_first_genom_file;
}  // namespace

Type StringToType(std::string str) {
  if (str == "half")
    return HALF;
  if (str == "convert_float")
    return CONVERT_FLOAT;
  return FLOAT;
}

int StringToInt(std::string str) {
  return std::stoi(str);
}

float StringToFloat(std::string str) {
  return std::stof(str);
}

bool StringToBool(std::string str) {
  if (str == "true")
    return true;
  else
    return false;
}

void SetFlag(std::string str, flags_type& flags) {
  std::string::size_type equal_pos = str.find("=");
  if (equal_pos == std::string::npos || (str[0] != '-' && str[1] != '-')) {
    std::cerr << "Flag Syntax Error: Cannot parse flags." << std::endl;
    std::cerr << "===Usage===\n ./bin/cnn MODE --flag1=hoge --flag2==fuga" << std::endl;
    exit(1);
  }
  std::string flag_name = std::string(str.begin() + 2, str.begin() + equal_pos);
  std::string flag_value = std::string(str.begin() + equal_pos + 1, str.end());

  flags_type::iterator it = flags.find(flag_name);
  if (it == flags.end()) {
    std::cerr << "Unknown Flag: \'" << flag_name << "\' is undefined." << std::endl;
  } else {
    it->second(flag_value);
  }
};

void Options::ParseCommandLine(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Please set mode(train/test)." << std::endl;
    exit(1);
  }

  std::string program = argv[0];
  std::string mode = (program != "./bin/cnn") ? "ga" : argv[1];
  if (mode == "train") {
    g_train = true;
  } else if (mode == "test") {
    if (argc < 3 || argv[2][0] == '-') {
      std::cerr << "Please set a weights input file." << std::endl;
      exit(1);
    }
    g_train=false;
    g_weights_input = argv[2];
  } else if (mode == "ga") {
    g_train=false;
    g_first_genom_file = argv[1];
  } else {
    std::cerr << "Please set mode(train/test)." << std::endl;
    exit(1);
  }
  
  flags_type flags;
  flags.insert(std::make_pair("type", [](std::string flag_value) {
        g_type = StringToType(flag_value);}));
  flags.insert(std::make_pair("exponent", [](std::string flag_value) {
        g_expoent = StringToInt(flag_value);}));
  flags.insert(std::make_pair("mantissa", [](std::string flag_value) {
        g_mantissa = StringToInt(flag_value);}));
  flags.insert(std::make_pair("weights_output", [](std::string flag_value) {
        g_weights_output = flag_value;
        g_save_params = true; }));
  flags.insert(std::make_pair("arithmatic_output", [](std::string flag_value) {
        g_arithmatic_output = flag_value;
        g_save_arithmetic = true; }));
  flags.insert(std::make_pair("cross_rate", [](std::string flag_value) {
        g_cross_rate = StringToFloat(flag_value); }));
  flags.insert(std::make_pair("mutation_rate", [](std::string flag_value) {
        g_mutation_rate = StringToFloat(flag_value); }));
  flags.insert(std::make_pair("max_generation", [](std::string flag_value) {
        g_max_generation = StringToInt(flag_value);}));
  for (int i = (mode == "test") ? 3 : 2; i < argc; ++i)
    SetFlag(argv[i], flags);
}

bool Options::IsTrain() {
  return g_train;
}

bool Options::IsSaveParams() {
  return g_save_params;
}

bool Options::IsSaveArithmetic() {
  return g_save_arithmetic;
}

Type Options::GetType() {
  return g_type;
}

int Options::GetExponent() {
  return g_expoent;
}

int Options::GetMantissa() {
  return g_mantissa;
}

std::string Options::GetWeightsInput() {
  return g_weights_input;
}

std::string Options::GetWeightsOutput() {
  return g_weights_output;
}

std::string Options::GetArithmaticOutput() {
  return g_arithmatic_output;
}

float Options::GetCrossRate() {
  return g_cross_rate;
}

float Options::GetMutationRate() {
  return g_mutation_rate;
}

int Options::GetMaxGeneration() {
  return g_max_generation;
}

std::string Options::GetFirstGenomFile() {
  return g_first_genom_file;
}
