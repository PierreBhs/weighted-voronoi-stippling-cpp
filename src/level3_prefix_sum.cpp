#include "level3_prefix_sum.hpp"

#include "spatial_grid.hpp"

namespace stippling {
namespace {

struct span_run
{
    std::uint32_t generator{};
    std::size_t   row{};
    std::size_t   x_begin{};
    std::size_t   x_end{};
};

struct prefix_tables
{
    std::size_t         width{};
    std::size_t         height{};
    std::vector<double> prefix_density;
    std::vector<double> prefix_x_density;
};

auto build_prefix_tables(std::span<const float> density, const std::size_t width, const std::size_t height)
    -> prefix_tables
{
    const auto size = width * height;

    auto prefix_density = std::vector<double>(size);
    auto prefix_x_density = std::vector<double>(size);

    for (auto y = 0uz; y < height; ++y) {
        const auto row = y * width;
        auto       sum_density = 0.;
        auto       sum_x_density = 0.;
        for (auto x = 0uz; x < width; ++x) {
            const auto d = static_cast<double>(density[row + x]);
            sum_density += d;
            sum_x_density += static_cast<double>(x) * d;
            prefix_density[row + x] = sum_density;
            prefix_x_density[row + x] = sum_x_density;
        }
    }

    return {width, height, std::move(prefix_density), std::move(prefix_x_density)};
}

void assign_voronoi_grid_with_spans(const spatial_grid&      grid,
                                    std::span<const vec2>    generators,
                                    std::span<std::uint32_t> voronoi,
                                    std::vector<span_run>&   spans,
                                    const std::size_t        width,
                                    const std::size_t        height)
{
    spans.clear();

    for (auto y = 0uz; y < height; ++y) {
        auto current_gen = std::uint32_t{0};
        auto span_start = 0uz;
        auto first_pixel = true;

        for (auto x = 0uz; x < width; ++x) {
            const auto best_idx = nearest_in_grid(grid, generators, static_cast<float>(x), static_cast<float>(y));
            voronoi[(y * width) + x] = best_idx;

            if (first_pixel) {
                current_gen = best_idx;
                span_start = x;
                first_pixel = false;
            } else if (best_idx != current_gen) {
                spans.push_back({
                    .generator = current_gen,
                    .row = y,
                    .x_begin = span_start,
                    .x_end = x - 1,
                });
                current_gen = best_idx;
                span_start = x;
            }
        }

        if (!first_pixel) {
            spans.push_back({
                .generator = current_gen,
                .row = y,
                .x_begin = span_start,
                .x_end = width - 1,
            });
        }
    }
}

void compute_centroids_prefix(std::span<const span_run> spans,
                              const prefix_tables&      tables,
                              std::span<accumulator>    accum)
{
    std::ranges::fill(accum, accumulator{});

    for (const auto& span : spans) {
        const auto row = span.row * tables.width;
        const auto fy = static_cast<double>(span.row);
        const auto right = row + span.x_end;
        const auto p_right = tables.prefix_density[right];
        const auto rx_right = tables.prefix_x_density[right];

        auto p_left = 0.;
        auto rx_left = 0.;
        if (span.x_begin > 0) {
            const auto left = row + span.x_begin - 1;
            p_left = tables.prefix_density[left];
            rx_left = tables.prefix_x_density[left];
        }

        const auto span_mass = p_right - p_left;
        accum[span.generator].mass += span_mass;
        accum[span.generator].moment_y += fy * span_mass;
        accum[span.generator].moment_x += rx_right - rx_left;
    }
}

}  // namespace

auto run_level3(const config& cfg, const image_data& image, const execution_options& exec_options)
    -> result<level_summary>
{
    auto grid = make_spatial_grid(image.width, image.height, cfg.num_generators);
    auto voronoi = std::vector<std::uint32_t>(image.width * image.height);
    auto spans = std::vector<span_run>{};
    auto tables = build_prefix_tables(image.density, image.width, image.height);

    spans.reserve(image.height *
                  static_cast<std::size_t>(std::max(16.0, std::sqrt(static_cast<double>(cfg.num_generators)) * 4.0)));

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
        assign_voronoi_grid_with_spans(grid, *generators, voronoi, spans, image.width, image.height);
        compute_centroids_prefix(spans, tables, accum);

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
            display_result(*generators, image.width, image.height, "Level 3: Prefix-Sum Centroids");
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

#if defined(STIPPLING_BUILD_LEVEL3_MAIN)
int main(int argc, char** argv)
{
    return stippling::run_cli_level(argc, argv, stippling::run_level3);
}
#endif
