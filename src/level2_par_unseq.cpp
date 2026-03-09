#include "level2_par_unseq.hpp"

#include "spatial_grid.hpp"

namespace stippling {

auto run_level2_par_unseq(const config& cfg, const image_data& image, const execution_options& exec_options)
    -> result<level_summary>
{
    auto grid = make_spatial_grid(image.width, image.height, cfg.num_generators);
    auto voronoi =
        std::vector<std::uint32_t>(image.width * image.height);

    auto generators = rejection_sample(image.density, image.width, image.height, cfg.num_generators);
    if (!generators) {
        return std::unexpected(generators.error());
    }

    auto accum = std::vector<accumulator>(cfg.num_generators);

    const auto total_t0 = steady_clock::now();
    auto       iterations_executed = 0uz;
    auto       converged = false;

    for (auto iter = 0uz; iter < cfg.max_iterations; ++iter) {
        populate_grid(grid, *generators);
        assign_voronoi_grid_par(grid, *generators, voronoi, image.width, image.height);
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
            display_result(*generators, image.width, image.height, "Level 2 par_unseq: Parallel Spatial Grid");
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

#if defined(STIPPLING_BUILD_LEVEL2_PAR_UNSEQ_MAIN)
int main(int argc, char** argv)
{
    return stippling::run_cli_level(argc, argv, stippling::run_level2_par_unseq);
}
#endif
