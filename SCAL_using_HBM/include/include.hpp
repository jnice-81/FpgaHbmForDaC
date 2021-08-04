#pragma once

#include <xcl2.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <functional>

typedef float vecType;

// DATA_SIZE is in vecType, ie here it's 4 GB. Note that DATA_SIZE % (1024*16) == 0 should hold,
// for most computing calls here, because the computing functions don't perform range checks

//#define DATA_SIZE (1024*1024*1024)  
#define DATA_SIZE (1024*32*4)
//#define DEBUGVERBOSE
//#define DEBUG

#include "MemorySplit.hpp"
#include "Adapters.hpp"
#include "helpers.hpp"
#include "calc_functions.hpp"
#include "test_helpers.hpp"
