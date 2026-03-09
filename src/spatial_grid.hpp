#pragma once

#include "common.hpp"

#include <algorithm>
#include <cassert>
#include <execution>
#include <limits>
#include <ranges>

namespace stippling {

struct spatial_grid
{
    float                                   cell_size{};
    int                                     cols{};
    int                                     rows{};
    std::vector<std::vector<std::uint32_t>> cells;
};

inline auto make_spatial_grid(const std::size_t width, const std::size_t height, const std::size_t num_generators)
    -> spatial_grid
{
    // Dividing by 2.f still creates big enough cells and makes the algo faster
    const auto cell_size =
        static_cast<float>(std::max(width, height)) / (std::sqrt(static_cast<float>(num_generators)) * 2.f);
    const auto cols = static_cast<int>(std::ceil(static_cast<float>(width) / cell_size));
    const auto rows = static_cast<int>(std::ceil(static_cast<float>(height) / cell_size));

    return {
        .cell_size = cell_size,
        .cols = cols,
        .rows = rows,
        .cells = std::vector<std::vector<std::uint32_t>>(static_cast<std::size_t>(cols * rows)),
    };
}

inline void populate_grid(spatial_grid& grid, std::span<const vec2> generators)
{
    std::ranges::for_each(grid.cells, [](auto& cell) { cell.clear(); });

    const float inv_cell_size = 1.f / grid.cell_size;
    for (const auto [i, gen] : std::views::enumerate(generators)) {
        const auto cx = static_cast<int>(gen.x * inv_cell_size);
        const auto cy = static_cast<int>(gen.y * inv_cell_size);

        assert(cx >= 0 && cx < grid.cols);
        assert(cy >= 0 && cy < grid.rows);

        grid.cells[(cy * grid.cols) + cx].push_back(static_cast<std::uint32_t>(i));
    }
}

inline auto nearest_in_grid(const spatial_grid& grid, std::span<const vec2> generators, const float qx, const float qy)
    -> std::uint32_t
{
    const auto cell_row = static_cast<int>(qy / grid.cell_size);
    const auto cell_col = static_cast<int>(qx / grid.cell_size);

    // Pre-calculate valid bounds to eliminate branches in the loop
    // 5x5 block of cells
    const int min_y = std::max(0, cell_row - 2);
    const int max_y = std::min(grid.rows - 1, cell_row + 2);
    const int min_x = std::max(0, cell_col - 2);
    const int max_x = std::min(grid.cols - 1, cell_col + 2);

    auto best_dist = std::numeric_limits<float>::max();
    auto best_idx = std::uint32_t{0};

    auto y_range = std::views::iota(min_y, max_y + 1);
    auto x_range = std::views::iota(min_x, max_x + 1);

    for (const auto [ny, nx] : std::views::cartesian_product(y_range, x_range)) {
        const auto cell_idx = static_cast<std::size_t>(ny * grid.cols + nx);

        for (const auto gen_idx : grid.cells[cell_idx]) {
            const auto gx = generators[gen_idx].x - qx;
            const auto gy = generators[gen_idx].y - qy;

            const auto dist = gx * gx + gy * gy;

            if (dist < best_dist) {
                best_dist = dist;
                best_idx = gen_idx;
            }
        }
    }

    return best_idx;
}

inline void assign_voronoi_grid(const spatial_grid&      grid,
                                std::span<const vec2>    generators,
                                std::span<std::uint32_t> voronoi,
                                const std::size_t        width,
                                const std::size_t        height)
{
    for (auto y = 0uz; y < height; ++y) {
        for (auto x = 0uz; x < width; ++x) {
            voronoi[(y * width) + x] = nearest_in_grid(grid, generators, static_cast<float>(x), static_cast<float>(y));
        }
    }
}

inline void assign_voronoi_grid_par(const spatial_grid&      grid,
                                    std::span<const vec2>    generators,
                                    std::span<std::uint32_t> voronoi,
                                    const std::size_t        width,
                                    const std::size_t        height)
{
    const auto rows = std::views::iota(0uz, height);
    std::for_each(
        std::execution::par_unseq, rows.begin(), rows.end(), [&grid, generators, voronoi, width](std::size_t y) {
            for (auto x = 0uz; x < width; ++x) {
                voronoi[(y * width) + x] =
                    nearest_in_grid(grid, generators, static_cast<float>(x), static_cast<float>(y));
            }
        });
}

}  // namespace stippling
