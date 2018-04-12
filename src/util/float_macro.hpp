#pragma once

#include <string>

#include "protos/arithmatic.pb.h"
#include "util/flags.hpp"
#include "util/float_type.hpp"
#include "util/converter.hpp"

Arithmatic::One p;

template<typename T>
void SaveArithmetic(const std::string operator_, const T a, const T b,
                    const T ans, const char* file, int line) {
    Arithmatic::Calculation* c = p.add_calc();
    c->set_file(file);
    c->set_line(line);
    c->set_operator_(operator_);
    c->set_a(Converter::ToFloat(a));
    c->set_b(Converter::ToFloat(b));
    c->set_ans(Converter::ToFloat(ans));
}

template<typename T>
T multiple(const T a, const T b, const char* file, int line) {
  T a_, b_;
  if (Flags::GetType() == CONVERT_FLOAT) {
    int e = Flags::GetExponent();
    int m = Flags::GetMantissa();
    a_ = BitConverter(e, m, a);
    b_ = BitConverter(e, m, b);
  }
  else {
    a_ = a;
    b_ = b;
  }
  T ans = a_ * b_;
  if (Flags::IsSaveArithmetic()) {
    SaveArithmetic("*", a_, b_, ans, file, line);
  }
  return ans;
}

template<typename T>
T division(const T a, const T b, const char* file, int line) {
  T a_, b_;
  if (Flags::GetType() == CONVERT_FLOAT) {
    int e = Flags::GetExponent();
    int m = Flags::GetMantissa();
    a_ = BitConverter(e, m, a);
    b_ = BitConverter(e, m, b);
  }
  else {
    a_ = a;
    b_ = b;
  }
  T ans = a_ / b_;
  if (Flags::IsSaveArithmetic()) {
    SaveArithmetic("/", a_, b_, ans, file, line);
  }
  return ans;
}

template<typename T>
T add(const T a, const T b, const char* file, int line) {
  T a_, b_;
  if (Flags::GetType() == CONVERT_FLOAT) {
    int e = Flags::GetExponent();
    int m = Flags::GetMantissa();
    a_ = BitConverter(e, m, a);
    b_ = BitConverter(e, m, b);
  }
  else {
    a_ = a;
    b_ = b;
  }
  T ans = a_ + b_;
  if (Flags::IsSaveArithmetic()) {
    SaveArithmetic("+", a_, b_, ans, file, line);
  }
  return ans;
}

template<typename T>
T sub(const T a, const T b, const char* file, int line) {
  T a_, b_;
  if (Flags::GetType() == CONVERT_FLOAT) {
    int e = Flags::GetExponent();
    int m = Flags::GetMantissa();
    a_ = BitConverter(e, m, a);
    b_ = BitConverter(e, m, b);
  }
  else {
    a_ = a;
    b_ = b;
  }
  T ans = a_ - b_;
  if (Flags::IsSaveArithmetic()) {
    SaveArithmetic("-", a_, b_, ans, file, line);
  }
  return ans;
}

#define MUL(a, b) multiple(a, b, __FILE__, __LINE__)
#define DIV(a, b) division(a, b, __FILE__, __LINE__)
#define ADD(a, b) add(a, b, __FILE__, __LINE__)
#define SUB(a, b) sub(a, b, __FILE__, __LINE__)

