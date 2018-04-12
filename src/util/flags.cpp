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
}  // namespace

bool StringToBool(std::string str) {
  if (str == "true")
    return true;
  return false;
}

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

void SetFlag(std::string str, flags_type& flags) {
  std::string::size_type equal_pos = str.find("=");
  if (equal_pos == std::string::npos || (str[0] != '-' && str[1] != '-')) {
    std::cerr << "Flag Syntax Error: Cannot parse flags." << std::endl;
    std::cerr << "===Usage===\n ./bin/cnn --flag1=hoge --flag2==fuga" << std::endl;
  }
  std::string flag_name = std::string(str.begin() + 2, str.begin() + equal_pos);
  std::string flag_value = std::string(str.begin() + equal_pos + 1, str.end());


    
  flags_type::iterator it = flags.find(flag_name);
  if (it == flags.end())
    std::cerr << "Unknown Flag: \'" << flag_name << "\' is undefined." << std::endl;
  else
    it->second(flag_value);
};

void Flags::ParseCommandLine(int argc, char* argv[]) {
  flags_type flags;
  flags.insert(std::make_pair("train", [](std::string flag_value) {
        g_train = StringToBool(flag_value);}));
  flags.insert(std::make_pair("save_params", [](std::string flag_value){
        g_save_params = StringToBool(flag_value);}));
  flags.insert(std::make_pair("save_arithmetic", [](std::string flag_value){
        g_save_arithmetic = StringToBool(flag_value);}));
  flags.insert(std::make_pair("type", [](std::string flag_value){
        g_type = StringToType(flag_value);}));
  flags.insert(std::make_pair("exponent", [](std::string flag_value){
        g_expoent = StringToInt(flag_value);}));
  flags.insert(std::make_pair("mantissa", [](std::string flag_value){
        g_mantissa = StringToInt(flag_value);}));
  
  for (int i = 1; i < argc; ++i) {
    SetFlag(argv[i], flags);
  }
}

bool Flags::IsTrain() {
  return g_train;
}

bool Flags::IsSaveParams() {
  return g_save_params;
}

bool Flags::IsSaveArithmetic() {
  return g_save_arithmetic;
}

Type Flags::GetType() {
  return g_type;
}

int Flags::GetExponent() {
  return g_expoent;
}

int Flags::GetMantissa() {
  return g_mantissa;
}
