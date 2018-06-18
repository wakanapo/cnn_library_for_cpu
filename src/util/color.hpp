#pragma once
#include <cstdlib>
#include <sstream>

enum Color {
  BLACK,
  RED,
  GREEN,
  YELLOW,
  BLUE,
  MAGENTA,
  CYAN,
  WHITE,
  DUMMY,
  NORMAL,
};

namespace {
  std::string textcolor({0x1b, '[', '1', ';', '3'});
  std::string getColorCode(Color c) {
    std::stringstream ss;
    ss << textcolor << c << "m";
    return ss.str();
  }
}  // namespace


std::string coloringText(std::string str, Color c) {
  std::stringstream ss;
  ss << getColorCode(c) << str << getColorCode(NORMAL);
  return ss.str();
}
