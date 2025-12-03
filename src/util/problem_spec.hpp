#pragma once

#include <cmath>
#include <fstream>
#include <vector>

using IC_Func = double (*)(double, double, double);

struct ProblemSpec {
    int N;                     // Number of grid points in each dimension
    double T;                  // Total simulation time
    IC_Func initial_condition; // Function pointer for initial condition
};

enum class Mode { profile, output, eval };

enum class Kernel { slow, fast };
