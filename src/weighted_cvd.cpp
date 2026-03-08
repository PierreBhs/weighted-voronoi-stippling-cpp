#include "common.hpp"
#include "spatial_grid.hpp"

#include <algorithm>

namespace stippling {

struct stippling_state
{
    std::vector<vec2>          generators;
    std::vector<accumulator>   accum;
    std::vector<std::uint32_t> voronoi;
    spatial_grid               grid;
    bool                       converged = false;
    std::size_t                iteration = 0;
    double                     last_displacement = 0.;
    double                     last_iteration_ms = 0.;
};

auto initialize(const config& cfg, const image_data& image) -> result<stippling_state>
{
    auto generators = rejection_sample(image.density, image.width, image.height, cfg.num_generators);
    if (!generators) {
        return std::unexpected(generators.error());
    }

    return stippling_state{
        .generators = std::move(*generators),
        .accum = std::vector<accumulator>(cfg.num_generators),
        .voronoi =
            std::vector<std::uint32_t>(static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height)),
        .grid = make_spatial_grid(image.width, image.height, cfg.num_generators),
        .converged = false,
        .iteration = 0,
        .last_displacement = 0.,
        .last_iteration_ms = 0.,
    };
}

void reset(stippling_state& state, const config& cfg, const image_data& image)
{
    auto generators = rejection_sample(image.density, image.width, image.height, cfg.num_generators);
    if (!generators) {
        return;
    }

    state.generators = std::move(*generators);
    std::ranges::fill(state.accum, accumulator{});
    state.iteration = 0;
    state.last_displacement = 0.;
    state.last_iteration_ms = 0.;
    state.converged = false;
}

void run_iteration(stippling_state& state, const config& cfg, const image_data& image)
{
    if (state.converged || state.iteration >= cfg.max_iterations) {
        return;
    }

    const auto iter_t0 = steady_clock::now();

    populate_grid(state.grid, state.generators);
    assign_voronoi_grid_par(state.grid, state.generators, state.voronoi, image.width, image.height);

    compute_centroids(state.voronoi, image.density, state.accum, image.width, image.height);
    const auto move = move_generators(state.generators, state.accum, image.width, image.height);

    state.last_iteration_ms = duration_ms(steady_clock::now() - iter_t0).count();
    state.iteration += 1;
    state.last_displacement = move.average_displacement;

    std::println("iteration {:3}  {:.1f} ms  displacement {:.4f}",
                 state.iteration,
                 state.last_iteration_ms,
                 state.last_displacement);

    if (move.average_displacement < cfg.convergence) {
        state.converged = true;
    }
}

void draw(const stippling_state& state)
{
    rl::BeginDrawing();
    rl::ClearBackground(rl::WHITE);

    for (const auto& generator : state.generators) {
        rl::DrawCircleV({generator.x, generator.y}, 1.5f, rl::BLACK);
    }

    rl::EndDrawing();
}

}  // namespace stippling

int main(int argc, char** argv)
{
    auto cfg = stippling::parse_args(argc, argv);
    if (!cfg) {
        std::println(stderr, "Error: {}", cfg.error());
        return 1;
    }

    auto image = stippling::load_image(cfg->image_path);
    if (!image) {
        std::println(stderr, "Error: {}", image.error());
        return 1;
    }

    auto state = stippling::initialize(*cfg, *image);
    if (!state) {
        std::println(stderr, "Error: {}", state.error());
        return 1;
    }

    rl::InitWindow(image->width, image->height, "Weighted Voronoi Stippling");
    rl::SetTargetFPS(20);

    while (!rl::WindowShouldClose()) {
        if (rl::IsKeyPressed(rl::KEY_R)) {
            stippling::reset(*state, *cfg, *image);
        }

        stippling::run_iteration(*state, *cfg, *image);
        stippling::draw(*state);
    }

    rl::CloseWindow();
    return 0;
}
