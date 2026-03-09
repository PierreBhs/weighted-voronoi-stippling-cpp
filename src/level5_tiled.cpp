#include "level5_tiled.hpp"

#include "spatial_grid.hpp"

#include <limits>
#include <numeric>

namespace stippling {
namespace {

struct tile_window
{
    std::size_t expanded_x0{};
    std::size_t expanded_y0{};
    std::size_t expanded_x1{};
    std::size_t expanded_y1{};
};

struct tiled_state
{
    spatial_grid               global_grid;
    spatial_grid               local_grid;
    std::vector<std::uint32_t> candidate_marks;
    std::uint32_t              mark_value = 1;
    std::vector<std::uint32_t> candidates;
    std::vector<vec2>          local_generators;
};

auto next_mark_value(tiled_state& state) -> std::uint32_t
{
    if (state.mark_value == std::numeric_limits<std::uint32_t>::max()) {
        std::ranges::fill(state.candidate_marks, std::uint32_t{0});
        state.mark_value = 1;
    }
    return state.mark_value++;
}

auto collect_tile_candidates(tiled_state&          state,
                             std::span<const vec2> generators,
                             const std::size_t     width,
                             const std::size_t     height,
                             const std::size_t     x0,
                             const std::size_t     y0,
                             const std::size_t     x1,
                             const std::size_t     y1,
                             const std::size_t     overlap) -> tile_window
{
    state.candidates.clear();

    const auto expanded_x0 = x0 > overlap ? x0 - overlap : 0uz;
    const auto expanded_y0 = y0 > overlap ? y0 - overlap : 0uz;
    const auto expanded_x1 = std::min(width, x1 + overlap);
    const auto expanded_y1 = std::min(height, y1 + overlap);

    const auto min_cell_x =
        std::clamp(static_cast<int>(std::floor(static_cast<float>(expanded_x0) / state.global_grid.cell_size)),
                   0,
                   state.global_grid.cols - 1);
    const auto min_cell_y =
        std::clamp(static_cast<int>(std::floor(static_cast<float>(expanded_y0) / state.global_grid.cell_size)),
                   0,
                   state.global_grid.rows - 1);
    const auto max_probe_x = expanded_x1 > 0 ? expanded_x1 - 1 : 0uz;
    const auto max_probe_y = expanded_y1 > 0 ? expanded_y1 - 1 : 0uz;
    const auto max_cell_x =
        std::clamp(static_cast<int>(std::floor(static_cast<float>(max_probe_x) / state.global_grid.cell_size)),
                   0,
                   state.global_grid.cols - 1);
    const auto max_cell_y =
        std::clamp(static_cast<int>(std::floor(static_cast<float>(max_probe_y) / state.global_grid.cell_size)),
                   0,
                   state.global_grid.rows - 1);

    const auto mark = next_mark_value(state);
    for (auto cell_y = min_cell_y; cell_y <= max_cell_y; ++cell_y) {
        for (auto cell_x = min_cell_x; cell_x <= max_cell_x; ++cell_x) {
            const auto& cell =
                state.global_grid.cells[static_cast<std::size_t>(cell_y * state.global_grid.cols + cell_x)];
            for (const auto gen_idx : cell) {
                if (state.candidate_marks[gen_idx] != mark) {
                    state.candidate_marks[gen_idx] = mark;
                    state.candidates.push_back(gen_idx);
                }
            }
        }
    }

    if (state.candidates.empty()) {
        state.candidates.resize(generators.size());
        std::iota(state.candidates.begin(), state.candidates.end(), std::uint32_t{0});
    }

    return tile_window{
        .expanded_x0 = expanded_x0,
        .expanded_y0 = expanded_y0,
        .expanded_x1 = expanded_x1,
        .expanded_y1 = expanded_y1,
    };
}

}  // namespace

auto run_level5(const config& cfg, const image_data& image, const execution_options& exec_options)
    -> result<level_summary>
{
    auto state = tiled_state{
        .global_grid = make_spatial_grid(image.width, image.height, cfg.num_generators),
        .local_grid = make_spatial_grid(1, 1, 1),
        .candidate_marks = std::vector<std::uint32_t>(cfg.num_generators, 0),
        .mark_value = 1,
        .candidates = {},
        .local_generators = {},
    };

    auto generators = rejection_sample(image.density, image.width, image.height, cfg.num_generators);
    if (!generators) {
        return std::unexpected(generators.error());
    }

    auto accum = std::vector<accumulator>(cfg.num_generators);

    const auto supersample = cfg.supersample;
    const auto sample_scale = 1. / static_cast<double>(supersample * supersample);
    const auto tile_step = static_cast<std::size_t>(cfg.tile_size);
    const auto overlap = static_cast<std::size_t>(cfg.tile_overlap);

    const auto total_t0 = steady_clock::now();
    auto       iter = 0uz;
    auto       converged = false;

    for (; iter < cfg.max_iterations; ++iter) {
        populate_grid(state.global_grid, *generators);
        std::ranges::fill(accum, accumulator{});

        for (auto tile_y0 = 0uz; tile_y0 < image.height; tile_y0 += tile_step) {
            const auto tile_y1 = std::min(tile_y0 + tile_step, image.height);
            for (auto tile_x0 = 0uz; tile_x0 < image.width; tile_x0 += tile_step) {
                const auto tile_x1 = std::min(tile_x0 + tile_step, image.width);
                const auto window = collect_tile_candidates(
                    state, *generators, image.width, image.height, tile_x0, tile_y0, tile_x1, tile_y1, overlap);

                state.local_generators.clear();
                state.local_generators.reserve(state.candidates.size());
                for (const auto gen_idx : state.candidates) {
                    state.local_generators.push_back({
                        (*generators)[gen_idx].x - static_cast<float>(window.expanded_x0),
                        (*generators)[gen_idx].y - static_cast<float>(window.expanded_y0),
                    });
                }

                state.local_grid = make_spatial_grid(window.expanded_x1 - window.expanded_x0,
                                                     window.expanded_y1 - window.expanded_y0,
                                                     state.local_generators.size());
                populate_grid(state.local_grid, state.local_generators);

                for (auto y = tile_y0; y < tile_y1; ++y) {
                    for (auto x = tile_x0; x < tile_x1; ++x) {
                        const auto density = static_cast<double>(image.density[(y * image.width) + x]);
                        if (density <= 0.) {
                            continue;
                        }

                        const auto sample_mass = density * sample_scale;
                        for (auto sy = 0; sy < supersample; ++sy) {
                            for (auto sx = 0; sx < supersample; ++sx) {
                                const auto qx = static_cast<float>(x) +
                                                (static_cast<float>(sx) + 0.5f) / static_cast<float>(supersample);
                                const auto qy = static_cast<float>(y) +
                                                (static_cast<float>(sy) + 0.5f) / static_cast<float>(supersample);
                                const auto local_owner = nearest_in_grid(state.local_grid,
                                                                         state.local_generators,
                                                                         qx - static_cast<float>(window.expanded_x0),
                                                                         qy - static_cast<float>(window.expanded_y0));
                                const auto owner = state.candidates[local_owner];
                                accum[owner].mass += sample_mass;
                                accum[owner].moment_x += static_cast<double>(qx) * sample_mass;
                                accum[owner].moment_y += static_cast<double>(qy) * sample_mass;
                            }
                        }
                    }
                }
            }
        }

        const auto move = move_generators(*generators, accum, image.width, image.height);
        if (move.average_displacement < cfg.convergence) {
            converged = true;
            ++iter;
            break;
        }
    }

    const auto total = duration_ms(steady_clock::now() - total_t0).count();

    if (exec_options.allow_presentation) {
        if (!cfg.output_path.empty()) {
            save_result(cfg.output_path, *generators, image.width, image.height);
        }
        if (cfg.display) {
            display_result(*generators, image.width, image.height, "Level 5: Tiled Supersampled Voronoi");
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

#if defined(STIPPLING_BUILD_LEVEL5_MAIN)
int main(int argc, char** argv)
{
    return stippling::run_cli_level(argc, argv, stippling::run_level5);
}
#endif
