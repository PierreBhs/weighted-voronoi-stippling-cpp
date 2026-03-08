#include "level4_par_unseq.hpp"

#include "quadtree.hpp"

#include <algorithm>
#include <execution>
#include <ranges>

namespace stippling {

auto run_level4_par_unseq(const config& cfg, const image_data& image, const execution_options& exec_options)
    -> result<level_summary>
{
    auto generators = rejection_sample(image.density, image.width, image.height, cfg.num_generators);
    if (!generators) {
        return std::unexpected(generators.error());
    }

    auto accum = std::vector<accumulator>(cfg.num_generators);
    auto tree = quadtree{};
    auto voronoi =
        std::vector<std::uint32_t>(static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height));

    const auto rows = std::views::iota(0, image.height);

    const auto total_t0 = steady_clock::now();
    auto       iterations_executed = 0uz;
    auto       converged = false;
    for (auto iter = 0uz; iter < cfg.max_iterations; ++iter) {
        tree.build(
            generators->data(), generators->size(), static_cast<float>(image.width), static_cast<float>(image.height));

        const auto w = image.width;
        std::for_each(std::execution::par_unseq, rows.begin(), rows.end(), [&tree, &voronoi, w](int y) {
            const auto row = static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
            const auto fy = static_cast<float>(y);
            for (auto x = 0; x < w; ++x) {
                voronoi[row + static_cast<std::size_t>(x)] = tree.nearest(static_cast<float>(x), fy);
            }
        });

        compute_centroids(voronoi, image.density, accum, image.width, image.height);

        const auto move = move_generators(*generators, accum, image.width, image.height);
        iterations_executed = iter + 1;

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
            display_result(*generators, image.width, image.height, "Level 4 par_unseq: Parallel STL Quadtree");
        }
    }

    return level_summary{
        .total_ms = total,
        .iterations_executed = iterations_executed,
        .converged = converged,
        .generators = std::move(*generators),
    };
}

}  // namespace stippling

#if defined(STIPPLING_BUILD_LEVEL4_PAR_UNSEQ_MAIN)
int main(int argc, char** argv)
{
    return stippling::run_cli_level(argc, argv, stippling::run_level4_par_unseq);
}
#endif
