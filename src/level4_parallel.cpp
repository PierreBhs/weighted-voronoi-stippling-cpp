#include "level4_parallel.hpp"

#include "quadtree.hpp"

#include <thread>

namespace stippling {
namespace {

auto worker_count() -> unsigned
{
    const auto hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 2;
}

void assign_voronoi_parallel(const quadtree&          tree,
                             std::span<std::uint32_t> voronoi,
                             const std::size_t        width,
                             const std::size_t        height,
                             const unsigned           num_workers)
{
    const auto rows_per_worker = (height + num_workers - 1) / num_workers;

    // Each thread writes to disjoint row ranges -- no synchronization needed
    auto workers = std::vector<std::jthread>{};
    workers.reserve(num_workers);

    for (auto t = 0u; t < num_workers; ++t) {
        const auto y_begin = static_cast<std::size_t>(t) * rows_per_worker;
        const auto y_end = std::min(y_begin + rows_per_worker, height);
        if (y_begin >= height) {
            break;
        }

        workers.emplace_back([&tree, voronoi, width, y_begin, y_end] {
            for (auto y = y_begin; y < y_end; ++y) {
                const auto fy = static_cast<float>(y);
                for (auto x = 0uz; x < width; ++x) {
                    voronoi[(y * width) + x] = tree.nearest(static_cast<float>(x), fy);
                }
            }
        });
    }
}

void compute_centroids_parallel(std::span<const std::uint32_t>         voronoi,
                                std::span<const float>                 density,
                                std::span<accumulator>                 accum,
                                std::vector<std::vector<accumulator>>& thread_accums,
                                const std::size_t                      width,
                                const std::size_t                      height,
                                const unsigned                         num_workers)
{
    const auto num_generators = accum.size();
    for (auto& local : thread_accums) {
        std::ranges::fill(local, accumulator{});
    }

    const auto rows_per_worker = (height + num_workers - 1) / num_workers;

    {
        auto workers = std::vector<std::jthread>{};
        workers.reserve(num_workers);

        for (auto t = 0u; t < num_workers; ++t) {
            const auto y_begin = static_cast<std::size_t>(t) * rows_per_worker;
            const auto y_end = std::min(y_begin + rows_per_worker, height);
            if (y_begin >= height) {
                break;
            }

            workers.emplace_back([&voronoi, &density, &thread_accums, t, width, y_begin, y_end] {
                auto& local = thread_accums[t];
                for (auto y = y_begin; y < y_end; ++y) {
                    for (auto x = 0uz; x < width; ++x) {
                        const auto idx = (y * width) + x;
                        const auto gen = voronoi[idx];
                        const auto d = static_cast<double>(density[idx]);
                        local[gen].mass += d;
                        local[gen].moment_x += static_cast<double>(x) * d;
                        local[gen].moment_y += static_cast<double>(y) * d;
                    }
                }
            });
        }
    }

    std::ranges::fill(accum, accumulator{});
    for (const auto& local : thread_accums) {
        for (auto i = 0uz; i < num_generators; ++i) {
            accum[i].mass += local[i].mass;
            accum[i].moment_x += local[i].moment_x;
            accum[i].moment_y += local[i].moment_y;
        }
    }
}

}  // namespace

auto run_level4_parallel(const config& cfg, const image_data& image, const execution_options& exec_options)
    -> result<level_summary>
{
    auto generators = rejection_sample(image.density, image.width, image.height, cfg.num_generators);
    if (!generators) {
        return std::unexpected(generators.error());
    }

    auto accum = std::vector<accumulator>(cfg.num_generators);
    auto tree = quadtree{};
    auto voronoi =
        std::vector<std::uint32_t>(image.width * image.height);

    const auto num_workers = worker_count();
    auto       thread_accums =
        std::vector<std::vector<accumulator>>(num_workers, std::vector<accumulator>(cfg.num_generators));

    const auto total_t0 = steady_clock::now();
    auto       iter = 0uz;
    auto       converged = false;

    for (; iter < cfg.max_iterations; ++iter) {
        tree.build(
            generators->data(), generators->size(), static_cast<float>(image.width), static_cast<float>(image.height));

        assign_voronoi_parallel(tree, voronoi, image.width, image.height, num_workers);
        compute_centroids_parallel(
            voronoi, image.density, accum, thread_accums, image.width, image.height, num_workers);

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
            display_result(*generators, image.width, image.height, "Level 4 Parallel: jthread Quadtree");
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

#if defined(STIPPLING_BUILD_LEVEL4_PARALLEL_MAIN)
int main(int argc, char** argv)
{
    return stippling::run_cli_level(argc, argv, stippling::run_level4_parallel);
}
#endif
