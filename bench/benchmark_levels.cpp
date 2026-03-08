#include "level1_brute_force.hpp"
#include "level2_par_unseq.hpp"
#include "level2_spatial_grid.hpp"
#include "level3_prefix_sum.hpp"
#include "level4_par_unseq.hpp"
#include "level4_parallel.hpp"
#include "level4_quadtree.hpp"
#include "level5_tiled.hpp"

#include <benchmark/benchmark.h>
#include <functional>

namespace {

using stippling::config;
using stippling::execution_options;
using stippling::image_data;

auto load_cached_image(const std::string& path) -> stippling::result<std::reference_wrapper<const image_data>>
{
    static auto cache = std::vector<std::pair<std::string, image_data>>{};
    for (const auto& [cached_path, image] : cache) {
        if (cached_path == path) {
            return std::cref(image);
        }
    }

    auto image = stippling::load_image(path);
    if (!image) {
        return std::unexpected(image.error());
    }
    cache.emplace_back(path, std::move(*image));
    return std::cref(cache.back().second);
}

template <typename Runner>
void run_benchmark(benchmark::State&  state,
                   const std::string& image_path,
                   Runner&&           runner,
                   const std::size_t  generators,
                   const std::size_t  iterations,
                   const int          tile_size = 256,
                   const int          tile_overlap = 64,
                   const int          supersample = 2)
{
    auto image = load_cached_image(image_path);
    if (!image) {
        state.SkipWithError(image.error().c_str());
        return;
    }

    auto cfg = config{};
    cfg.image_path = image_path;
    cfg.num_generators = generators;
    cfg.max_iterations = iterations;
    cfg.display = false;
    cfg.output_path.clear();
    cfg.tile_size = tile_size;
    cfg.tile_overlap = tile_overlap;
    cfg.supersample = supersample;

    auto options = execution_options{
        .allow_presentation = false,
    };

    auto last_summary = stippling::level_summary{};
    for (auto _ : state) {
        auto summary = runner(cfg, image->get(), options);
        if (!summary) {
            state.SkipWithError(summary.error().c_str());
            return;
        }
        last_summary = std::move(*summary);
        benchmark::DoNotOptimize(last_summary.total_ms);
        benchmark::ClobberMemory();
    }

    state.counters["iters"] = static_cast<double>(last_summary.iterations_executed);
    state.counters["converged"] = last_summary.converged ? 1. : 0.;
    state.counters["total_ms"] = last_summary.total_ms;
    state.counters["ms_per_iter"] = last_summary.total_ms / static_cast<double>(last_summary.iterations_executed);
}

// --- Level 1: brute-force reference (small image, few generators) ---

void bench_level1_giraffe(benchmark::State& state)
{
    run_benchmark(state, "data/giraffe.jpg", stippling::run_level1, 1000, 20);
}
BENCHMARK(bench_level1_giraffe)->Name("level1/giraffe/1000")->Unit(benchmark::kMillisecond);

// --- giraffe.jpg (1024x683) ---

void bench_level2_giraffe(benchmark::State& state)
{
    run_benchmark(state, "data/giraffe.jpg", stippling::run_level2, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level2_giraffe)
    ->Name("level2/giraffe")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level3_giraffe(benchmark::State& state)
{
    run_benchmark(state, "data/giraffe.jpg", stippling::run_level3, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level3_giraffe)
    ->Name("level3/giraffe")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level4_giraffe(benchmark::State& state)
{
    run_benchmark(state, "data/giraffe.jpg", stippling::run_level4, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_giraffe)
    ->Name("level4/giraffe")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level5_giraffe(benchmark::State& state)
{
    run_benchmark(
        state, "data/giraffe.jpg", stippling::run_level5, 5000, 25, 256, 64, static_cast<int>(state.range(0)));
}
BENCHMARK(bench_level5_giraffe)->Name("level5/giraffe")->Arg(2)->Arg(3)->Unit(benchmark::kMillisecond);

// --- einstein.jpg (1920x1080) ---

void bench_level2_einstein(benchmark::State& state)
{
    run_benchmark(state, "data/einstein.jpg", stippling::run_level2, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level2_einstein)
    ->Name("level2/einstein")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level3_einstein(benchmark::State& state)
{
    run_benchmark(state, "data/einstein.jpg", stippling::run_level3, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level3_einstein)
    ->Name("level3/einstein")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level4_einstein(benchmark::State& state)
{
    run_benchmark(state, "data/einstein.jpg", stippling::run_level4, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_einstein)
    ->Name("level4/einstein")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

// --- david.png (2752x1536) ---

void bench_level2_david(benchmark::State& state)
{
    run_benchmark(state, "data/david.png", stippling::run_level2, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level2_david)->Name("level2/david")->Arg(5000)->Arg(10000)->Arg(20000)->Unit(benchmark::kMillisecond);

void bench_level3_david(benchmark::State& state)
{
    run_benchmark(state, "data/david.png", stippling::run_level3, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level3_david)->Name("level3/david")->Arg(5000)->Arg(10000)->Arg(20000)->Unit(benchmark::kMillisecond);

void bench_level4_david(benchmark::State& state)
{
    run_benchmark(state, "data/david.png", stippling::run_level4, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_david)->Name("level4/david")->Arg(5000)->Arg(10000)->Arg(20000)->Unit(benchmark::kMillisecond);

// --- Level 2 par_unseq: parallel spatial grid (all images) ---

void bench_level2_par_unseq_giraffe(benchmark::State& state)
{
    run_benchmark(
        state, "data/giraffe.jpg", stippling::run_level2_par_unseq, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level2_par_unseq_giraffe)
    ->Name("level2_par_unseq/giraffe")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level2_par_unseq_einstein(benchmark::State& state)
{
    run_benchmark(
        state, "data/einstein.jpg", stippling::run_level2_par_unseq, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level2_par_unseq_einstein)
    ->Name("level2_par_unseq/einstein")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level2_par_unseq_david(benchmark::State& state)
{
    run_benchmark(
        state, "data/david.png", stippling::run_level2_par_unseq, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level2_par_unseq_david)
    ->Name("level2_par_unseq/david")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

// --- Level 4 parallel: jthread quadtree (all images) ---

void bench_level4_parallel_giraffe(benchmark::State& state)
{
    run_benchmark(
        state, "data/giraffe.jpg", stippling::run_level4_parallel, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_parallel_giraffe)
    ->Name("level4_parallel/giraffe")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level4_parallel_einstein(benchmark::State& state)
{
    run_benchmark(
        state, "data/einstein.jpg", stippling::run_level4_parallel, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_parallel_einstein)
    ->Name("level4_parallel/einstein")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level4_parallel_david(benchmark::State& state)
{
    run_benchmark(
        state, "data/david.png", stippling::run_level4_parallel, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_parallel_david)
    ->Name("level4_parallel/david")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

// --- Level 4 par_unseq: parallel STL quadtree (all images) ---

void bench_level4_par_unseq_giraffe(benchmark::State& state)
{
    run_benchmark(
        state, "data/giraffe.jpg", stippling::run_level4_par_unseq, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_par_unseq_giraffe)
    ->Name("level4_par_unseq/giraffe")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level4_par_unseq_einstein(benchmark::State& state)
{
    run_benchmark(
        state, "data/einstein.jpg", stippling::run_level4_par_unseq, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_par_unseq_einstein)
    ->Name("level4_par_unseq/einstein")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

void bench_level4_par_unseq_david(benchmark::State& state)
{
    run_benchmark(
        state, "data/david.png", stippling::run_level4_par_unseq, static_cast<std::size_t>(state.range(0)), 50);
}
BENCHMARK(bench_level4_par_unseq_david)
    ->Name("level4_par_unseq/david")
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(20000)
    ->Unit(benchmark::kMillisecond);

}  // namespace
