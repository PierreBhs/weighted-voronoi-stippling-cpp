#include "level1_brute_force.hpp"

#include <limits>

namespace stippling {
namespace {

void assign_voronoi_brute_force(std::span<const vec2>    generators,
                                std::span<std::uint32_t> voronoi,
                                const std::size_t        width,
                                const std::size_t        height)
{
    const auto generator_count = static_cast<std::uint32_t>(generators.size());

    for (auto y = 0uz; y < height; ++y) {
        for (auto x = 0uz; x < width; ++x) {
            auto best_dist = std::numeric_limits<float>::max();
            auto best_idx = std::uint32_t{0};

            for (auto i = std::uint32_t{0}; i < generator_count; ++i) {
                const auto dx = generators[i].x - static_cast<float>(x);
                const auto dy = generators[i].y - static_cast<float>(y);
                const auto dist = dx * dx + dy * dy;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }

            voronoi[(y * width) + x] = best_idx;
        }
    }
}

}  // namespace

auto run_level1(const config& cfg, const image_data& image, const execution_options& exec_options)
    -> result<level_summary>
{
    auto generators = rejection_sample(image.density, image.width, image.height, cfg.num_generators);
    if (!generators) {
        return std::unexpected(generators.error());
    }

    auto accum = std::vector<accumulator>(cfg.num_generators);
    auto voronoi =
        std::vector<std::uint32_t>(image.width * image.height);

    const auto total_t0 = steady_clock::now();
    auto       iter = 0uz;
    auto       converged = false;

    for (; iter < cfg.max_iterations; ++iter) {
        assign_voronoi_brute_force(*generators, voronoi, image.width, image.height);
        compute_centroids(voronoi, image.density, accum, image.width, image.height);

        const auto move = move_generators(*generators, accum, image.width, image.height);
        if (move.average_displacement < cfg.convergence) {
            converged = true;
            break;
        }
    }

    const auto total = duration_ms(steady_clock::now() - total_t0).count();

    if (exec_options.allow_presentation) {
        if (!cfg.output_path.empty()) {
            save_result(cfg.output_path, *generators, image.width, image.height);
        }
        if (cfg.display) {
            display_result(*generators, image.width, image.height, "Level 1: Brute-Force Voronoi");
        }
    }

    return level_summary{
        .total_ms = total,
        .iterations_executed = iter,
        .converged = converged,
        .generators = std::move(*generators),
    };
}

}  // namespace stippling

#if defined(STIPPLING_BUILD_LEVEL1_MAIN)
int main(int argc, char** argv)
{
    return stippling::run_cli_level(argc, argv, stippling::run_level1);
}
#endif
