#pragma once

#include <uhdm/uhdm.h>

namespace cudalator {
class SurelogParser {
public:
    SurelogParser();

    void parse(const vpiHandle &handle);
};
}
