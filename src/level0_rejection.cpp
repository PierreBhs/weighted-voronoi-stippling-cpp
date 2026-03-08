#include "level0_rejection.hpp"

namespace stippling {

auto run_level0(const config& cfg, const image_data& image, const execution_options& exec_options)
    -> result<level_summary>
{
    const auto t0 = steady_clock::now();

    auto generators = rejection_sample(image.density, image.width, image.height, cfg.num_generators);
    if (!generators) {
        return std::unexpected(generators.error());
    }

    const auto total = duration_ms(steady_clock::now() - t0).count();

    if (exec_options.allow_presentation) {
        if (!cfg.output_path.empty()) {
            save_result(cfg.output_path, *generators, image.width, image.height);
        }
        if (cfg.display) {
            display_result(*generators, image.width, image.height, "Level 0: Rejection Sampling");
        }
    }

    return level_summary{
        .total_ms = total,
        .iterations_executed = 1,
        .converged = false,
        .generators = std::move(*generators),
    };
}

}  // namespace stippling

#if defined(STIPPLING_BUILD_LEVEL0_MAIN)
int main(int argc, char** argv)
{
    return stippling::run_cli_level(argc, argv, stippling::run_level0);
}
#endif
