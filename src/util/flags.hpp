#pragma once

enum Type {
  FLOAT,
  HALF,
  CONVERT_FLOAT
};

class Flags {
public:
  static void ParseCommandLine(int argc, char* argv[]);
  static bool IsTrain();
  static bool IsSaveParams();
  static bool IsSaveArithmetic();
  static Type GetType();
  static int GetExponent();
  static int GetMantissa();
};
