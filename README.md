# Pointy Pictures

C++23 implementations of *[Weighted Voronoi Stippling](https://www.cs.ubc.ca/labs/imager/tr/2002/secord2002b/secord.2002b.pdf)*

<video src="img/stipple_onepiece_video.mp4" autoplay loop muted playsinline></video>

The repo implements:

- `level0`: weighted rejection sampling baseline
- `level1`: brute-force Lloyd relaxation
- `level2`: spatial-grid Voronoi assignment
- `level3`: spatial-grid Voronoi + span-driven prefix-sum centroids
- `level4`: quadtree nearest-neighbor assignment
- `level5`: tiled supersampled Voronoi for higher-precision centroids

## Build

```sh
conan install . --build=missing
cmake --preset conan-release
cmake --build --preset conan-release
```

## Run

Multiple implementations, used for benchmarking. Main file for visualization is `src/weighted_cvd.cpp`, which you can run like:

```sh
./build/Release/weighted_cvd --image data/onepiece.jpg --generators 200000 --iterations 50
```

The other files are used for the benchmark but can also be executed, with these cli args:

- `--image <path>`
- `--generators <count>`
- `--iterations <count>`
- `--convergence <epsilon>`
- `--output <path>`
- `--display`

## Benchmarks

Benchmarks are implemented with Google Benchmark in `bench`.

Examples:

```sh
./build/Release/voronoi_bench --benchmark_filter='level2/.*|level3/.*|level4/.*'
./build/Release/voronoi_bench --benchmark_filter='level5/giraffe/2' --benchmark_min_time=0.01s
./build/Release/voronoi_bench --benchmark_out=bench.json --benchmark_out_format=json
```

Each benchmark row reports the benchmark-library timing plus user counters such as:

- `iters`: actual Lloyd iterations executed
- `converged`: whether the run converged before the requested limit
- `total_ms`: end-to-end algorithm time reported by the implementation itself
