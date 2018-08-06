#pragma once

#include <string>

enum Type {
  FLOAT,
  HALF,
  CONVERT_FLOAT
};

class Options {
public:
  static void ParseCommandLine(int argc, char* argv[]);
  static bool IsTrain();
  static bool IsSaveParams();
  static bool IsSaveArithmetic();
  static Type GetType();
  static int GetExponent();
  static int GetMantissa();
  static std::string GetWeightsInput();
  static std::string GetWeightsOutput();
  static std::string GetArithmaticOutput();
  static float GetCrossRate();
  static float GetMutationRate();
  static int GetMaxGeneration();
  static std::string GetFirstGenomFile();
};
