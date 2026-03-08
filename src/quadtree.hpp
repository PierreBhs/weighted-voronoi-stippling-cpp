#pragma once

#include "common.hpp"

#include <array>
#include <limits>
#include <numeric>

namespace stippling {

struct quadtree
{
    static constexpr std::size_t bucket_size = 16;
    static constexpr std::size_t max_depth = 20;

    struct node
    {
        float       x0{};
        float       y0{};
        float       x1{};
        float       y1{};
        std::size_t first_child{};
        std::size_t point_start{};
        std::size_t point_count{};
        bool        is_leaf{true};
    };

    std::vector<node>          nodes;
    std::vector<std::uint32_t> points;
    const vec2*                generators = nullptr;

    void build(const vec2* gens, const std::size_t count, const float width, const float height)
    {
        generators = gens;
        points.resize(count);
        std::iota(points.begin(), points.end(), std::uint32_t{0});

        nodes.clear();
        nodes.reserve(count * 2);
        nodes.push_back({0.f, 0.f, width, height, 0, 0, count, true});

        scratch_.clear();
        subdivide(0, 0);
    }

    [[nodiscard]] auto nearest(const float qx, const float qy) const -> std::uint32_t
    {
        auto best_dist = std::numeric_limits<float>::max();
        auto best_idx = std::uint32_t{0};
        nearest_impl(0, qx, qy, best_dist, best_idx);
        return best_idx;
    }

private:
    std::vector<std::uint32_t> scratch_;

    [[nodiscard]] auto quad_of(const float mx, const float my, const vec2& point) const -> std::size_t
    {
        return (point.x >= mx ? 1uz : 0uz) | (point.y >= my ? 2uz : 0uz);
    }

    void subdivide(const std::size_t node_index, const std::size_t depth)
    {
        if (nodes[node_index].point_count <= bucket_size || depth >= max_depth) {
            return;
        }

        const auto mx = (nodes[node_index].x0 + nodes[node_index].x1) * 0.5f;
        const auto my = (nodes[node_index].y0 + nodes[node_index].y1) * 0.5f;
        const auto start = nodes[node_index].point_start;
        const auto count = nodes[node_index].point_count;

        auto counts = std::array<std::size_t, 4>{};
        for (auto i = start; i < start + count; ++i) {
            counts[quad_of(mx, my, generators[points[i]])]++;
        }

        auto offsets = std::array<std::size_t, 4>{};
        offsets[0] = start;
        for (auto i = 1uz; i < offsets.size(); ++i) {
            offsets[i] = offsets[i - 1] + counts[i - 1];
        }

        scratch_.assign(points.begin() + static_cast<std::ptrdiff_t>(start),
                        points.begin() + static_cast<std::ptrdiff_t>(start + count));
        auto positions = offsets;
        for (const auto point_index : scratch_) {
            points[positions[quad_of(mx, my, generators[point_index])]++] = point_index;
        }

        const auto x0 = nodes[node_index].x0;
        const auto y0 = nodes[node_index].y0;
        const auto x1 = nodes[node_index].x1;
        const auto y1 = nodes[node_index].y1;
        const auto first_child = nodes.size();

        nodes[node_index].first_child = first_child;
        nodes[node_index].is_leaf = false;

        const auto bounds = std::array<std::array<float, 4>, 4>{
            std::array<float, 4>{x0, y0, mx, my},
            std::array<float, 4>{mx, y0, x1, my},
            std::array<float, 4>{x0, my, mx, y1},
            std::array<float, 4>{mx, my, x1, y1},
        };

        for (auto quadrant = 0uz; quadrant < bounds.size(); ++quadrant) {
            nodes.push_back({
                bounds[quadrant][0],
                bounds[quadrant][1],
                bounds[quadrant][2],
                bounds[quadrant][3],
                0,
                offsets[quadrant],
                counts[quadrant],
                true,
            });
        }

        for (auto quadrant = 0uz; quadrant < bounds.size(); ++quadrant) {
            subdivide(first_child + quadrant, depth + 1);
        }
    }

    void nearest_impl(const std::size_t node_index,
                      const float       qx,
                      const float       qy,
                      float&            best_dist,
                      std::uint32_t&    best_idx) const
    {
        const auto& current_node = nodes[node_index];
        const auto  dx = std::max(0.f, std::max(current_node.x0 - qx, qx - current_node.x1));
        const auto  dy = std::max(0.f, std::max(current_node.y0 - qy, qy - current_node.y1));
        if (dx * dx + dy * dy >= best_dist) {
            return;
        }

        if (current_node.is_leaf) {
            for (auto i = current_node.point_start; i < current_node.point_start + current_node.point_count; ++i) {
                const auto gen_idx = points[i];
                const auto gx = generators[gen_idx].x - qx;
                const auto gy = generators[gen_idx].y - qy;
                const auto dist = gx * gx + gy * gy;
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = gen_idx;
                }
            }
            return;
        }

        const auto first_child = current_node.first_child;
        const auto primary =
            quad_of((current_node.x0 + current_node.x1) * 0.5f, (current_node.y0 + current_node.y1) * 0.5f, {qx, qy});

        static constexpr std::size_t visit_order[4][4] = {
            {0, 1, 2, 3},
            {1, 0, 3, 2},
            {2, 3, 0, 1},
            {3, 2, 1, 0},
        };

        for (const auto quadrant : visit_order[primary]) {
            nearest_impl(first_child + quadrant, qx, qy, best_dist, best_idx);
        }
    }
};

}  // namespace stippling
