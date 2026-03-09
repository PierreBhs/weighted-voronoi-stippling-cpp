#pragma once

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rl {
#include <raylib.h>
}

namespace stippling {

struct vec2
{
    float x{};
    float y{};
};

struct accumulator
{
    double mass{};
    double moment_x{};
    double moment_y{};
};

struct move_result
{
    double average_displacement{};
};

template <typename T>
using result = std::expected<T, std::string>;

struct config
{
    std::string image_path = "data/giraffe.jpg";
    std::size_t num_generators = 5000;
    std::size_t max_iterations = 50;
    double      convergence = 0.01;
    bool        display = false;
    std::string output_path;
    int         tile_size = 256;
    int         tile_overlap = 64;
    int         supersample = 2;
};

struct image_data
{
    std::size_t        width{};
    std::size_t        height{};
    std::vector<float> density;
};

struct density_sampler
{
    std::size_t                             width{};
    std::size_t                             height{};
    std::discrete_distribution<std::size_t> pixel_distribution;

    auto sample_point(std::mt19937& rng) -> vec2
    {
        auto       jitter = std::uniform_real_distribution<float>{0.f, 0.999f};
        const auto idx = pixel_distribution(rng);
        const auto px = idx % width;
        const auto py = idx / width;
        return {
            static_cast<float>(px) + jitter(rng),
            static_cast<float>(py) + jitter(rng),
        };
    }
};

struct execution_options
{
    bool allow_presentation = true;
};

struct level_summary
{
    double            total_ms{};
    std::size_t       iterations_executed{};
    bool              converged{};
    std::vector<vec2> generators;
};

using steady_clock = std::chrono::steady_clock;
using duration_ms = std::chrono::duration<double, std::milli>;

template <typename T>
inline auto parse_number(const std::string_view text, const char* name) -> result<T>
{
    auto        value = T{};
    const auto* begin = text.data();
    const auto* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(begin, end, value);
    if (ec != std::errc{} || ptr != end) {
        return std::unexpected(std::format("invalid value for {}: '{}'", name, text));
    }
    return value;
}

inline auto validate_config(const config& cfg) -> result<void>
{
    if (cfg.num_generators == 0) {
        return std::unexpected("--generators must be greater than zero");
    }
    if (cfg.max_iterations == 0) {
        return std::unexpected("--iterations must be greater than zero");
    }
    if (cfg.convergence < 0.) {
        return std::unexpected("--convergence must be non-negative");
    }
    if (cfg.tile_size <= 0) {
        return std::unexpected("--tile-size must be greater than zero");
    }
    if (cfg.tile_overlap < 0) {
        return std::unexpected("--tile-overlap must be non-negative");
    }
    if (cfg.supersample <= 0) {
        return std::unexpected("--supersample must be greater than zero");
    }
    return {};
}

inline auto parse_args(const int argc, char** argv) -> result<config>
{
    auto cfg = config{};
    for (int i = 1; i < argc; ++i) {
        const auto arg = std::string_view(argv[i]);
        if (arg == "--display") {
            cfg.display = true;
        } else if (arg == "--image" && i + 1 < argc) {
            cfg.image_path = argv[++i];
        } else if (arg == "--generators" && i + 1 < argc) {
            auto value = parse_number<std::size_t>(argv[++i], "--generators");
            if (!value) {
                return std::unexpected(value.error());
            }
            cfg.num_generators = *value;
        } else if (arg == "--iterations" && i + 1 < argc) {
            auto value = parse_number<std::size_t>(argv[++i], "--iterations");
            if (!value) {
                return std::unexpected(value.error());
            }
            cfg.max_iterations = *value;
        } else if (arg == "--convergence" && i + 1 < argc) {
            auto value = parse_number<double>(argv[++i], "--convergence");
            if (!value) {
                return std::unexpected(value.error());
            }
            cfg.convergence = *value;
        } else if (arg == "--output" && i + 1 < argc) {
            cfg.output_path = argv[++i];
        } else if (arg == "--tile-size" && i + 1 < argc) {
            auto value = parse_number<int>(argv[++i], "--tile-size");
            if (!value) {
                return std::unexpected(value.error());
            }
            cfg.tile_size = *value;
        } else if (arg == "--tile-overlap" && i + 1 < argc) {
            auto value = parse_number<int>(argv[++i], "--tile-overlap");
            if (!value) {
                return std::unexpected(value.error());
            }
            cfg.tile_overlap = *value;
        } else if (arg == "--supersample" && i + 1 < argc) {
            auto value = parse_number<int>(argv[++i], "--supersample");
            if (!value) {
                return std::unexpected(value.error());
            }
            cfg.supersample = *value;
        } else {
            return std::unexpected(std::format("unknown or incomplete argument '{}'", arg));
        }
    }

    auto validation = validate_config(cfg);
    if (!validation) {
        return std::unexpected(validation.error());
    }
    return cfg;
}

[[nodiscard]] inline auto load_image(std::string_view path) -> result<image_data>
{
    rl::SetTraceLogLevel(rl::LOG_WARNING);
    auto image = rl::LoadImage(path.data());
    if (image.data == nullptr || image.width <= 0 || image.height <= 0) {
        return std::unexpected(std::format("failed to load image '{}'", path));
    }

    rl::ImageFormat(&image, rl::PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);

    const auto size = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height);
    const auto pixels = std::span(reinterpret_cast<const std::uint8_t*>(image.data), size);

    auto density = std::vector<float>(size);
    for (auto i = 0uz; i < size; ++i) {
        density[i] = 1.f - static_cast<float>(pixels[i]) / 255.f;
    }

    rl::UnloadImage(image);
    return image_data{
        static_cast<std::size_t>(image.width), static_cast<std::size_t>(image.height), std::move(density)};
}

[[nodiscard]] inline auto make_density_sampler(std::span<const float> density,
                                               const std::size_t      width,
                                               const std::size_t      height) -> density_sampler
{
    return density_sampler{
        .width = width,
        .height = height,
        .pixel_distribution = std::discrete_distribution<std::size_t>(density.begin(), density.end()),
    };
}

// [[nodiscard]] inline constexpr auto to_idx(auto x, auto y) -> std::size_t {}

inline auto rejection_sample(std::span<const float> density,
                             const std::size_t      width,
                             const std::size_t      height,
                             const std::size_t      count,
                             const std::uint32_t    seed = 2026) -> result<std::vector<vec2>>
{
    auto fallback_sampler = make_density_sampler(density, width, height);

    auto rng = std::mt19937{seed};
    auto dist_x = std::uniform_int_distribution<std::size_t>{0, width - 1};
    auto dist_y = std::uniform_int_distribution<std::size_t>{0, height - 1};
    auto dist_p = std::uniform_real_distribution<float>{0.f, 1.f};

    auto points = std::vector<vec2>{};
    points.reserve(count);

    auto           attempts_since_hit = 0uz;
    constexpr auto max_misses = 1'000'000uz;

    while (points.size() < count) {
        const auto x = dist_x(rng);
        const auto y = dist_y(rng);
        if (density[(y * width) + x] > dist_p(rng)) {
            points.emplace_back(static_cast<float>(x), static_cast<float>(y));
            attempts_since_hit = 0;
        } else if (++attempts_since_hit >= max_misses) {
            while (points.size() < count) {
                points.push_back(fallback_sampler.sample_point(rng));
            }
        }
    }

    return points;
}

inline void compute_centroids(std::span<const std::uint32_t> voronoi,
                              std::span<const float>         density,
                              std::span<accumulator>         accum,
                              const std::size_t              width,
                              const std::size_t              height)
{
    std::ranges::fill(accum, accumulator{});

    for (auto y = 0uz; y < height; ++y) {
        for (auto x = 0uz; x < width; ++x) {
            const auto idx = (y * width) + x;
            const auto gen = voronoi[idx];
            const auto d = static_cast<double>(density[idx]);
            accum[gen].mass += d;
            accum[gen].moment_x += static_cast<double>(x) * d;
            accum[gen].moment_y += static_cast<double>(y) * d;
        }
    }
}

inline auto move_generators(std::span<vec2>              generators,
                            std::span<const accumulator> accum,
                            const std::size_t            width,
                            const std::size_t            height) -> move_result
{
    // Sum squared displacements, take a single sqrt at the end (RMS).
    // This avoids N per-generator sqrt calls
    auto       sum_sq_displacement = 0.;
    const auto fw = static_cast<float>(width) - 0.001f;
    const auto fh = static_cast<float>(height) - 0.001f;

    for (auto i = 0uz; i < generators.size(); ++i) {
        if (accum[i].mass > 0.) {
            const auto new_x = std::clamp(static_cast<float>(accum[i].moment_x / accum[i].mass), 0.f, fw);
            const auto new_y = std::clamp(static_cast<float>(accum[i].moment_y / accum[i].mass), 0.f, fh);

            const auto dx = new_x - generators[i].x;
            const auto dy = new_y - generators[i].y;
            sum_sq_displacement += static_cast<double>(dx * dx + dy * dy);
            generators[i] = {new_x, new_y};
        }
    }

    return {
        .average_displacement = std::sqrt(sum_sq_displacement / static_cast<double>(generators.size())),
    };
}

inline void display_result(std::span<const vec2> generators,
                           const std::size_t     width,
                           const std::size_t     height,
                           const char*           title)
{
    rl::InitWindow(static_cast<int>(width), static_cast<int>(height), title);
    rl::SetTargetFPS(1);

    while (!rl::WindowShouldClose()) {
        rl::BeginDrawing();
        rl::ClearBackground(rl::WHITE);
        for (const auto& [x, y] : generators) {
            rl::DrawCircle(static_cast<int>(x), static_cast<int>(y), 1.f, rl::BLACK);
        }
        rl::EndDrawing();
    }

    rl::CloseWindow();
}

inline void save_result(std::string_view      path,
                        std::span<const vec2> generators,
                        const std::size_t     width,
                        const std::size_t     height)
{
    auto img = rl::GenImageColor(static_cast<int>(width), static_cast<int>(height), rl::WHITE);
    for (const auto& [x, y] : generators) {
        rl::ImageDrawCircle(&img, static_cast<int>(x), static_cast<int>(y), 1, rl::BLACK);
    }
    rl::ExportImage(img, path.data());
    rl::UnloadImage(img);
    std::println("Saved result to {}", path);
}

template <typename Runner>
inline int run_cli_level(int argc, char** argv, Runner&& runner)
{
    auto cfg = parse_args(argc, argv);
    if (!cfg) {
        std::println(stderr, "Error: {}", cfg.error());
        return 1;
    }

    auto image = load_image(cfg->image_path);
    if (!image) {
        std::println(stderr, "Error: {}", image.error());
        return 1;
    }

    auto summary = std::forward<Runner>(runner)(*cfg, *image, execution_options{});
    if (!summary) {
        std::println(stderr, "Error: {}", summary.error());
        return 1;
    }

    return 0;
}

}  // namespace stippling
