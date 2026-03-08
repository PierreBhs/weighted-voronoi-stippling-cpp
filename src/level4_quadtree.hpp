#pragma once

#include "common.hpp"

namespace stippling {

[[nodiscard]] auto run_level4(const config& cfg, const image_data& image, const execution_options& exec_options = {})
    -> result<level_summary>;

}  // namespace stippling
