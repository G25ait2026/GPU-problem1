# GPU-Accelerated 2D Heat Diffusion: From Jacobi and Red-Black SOR to Geometric Multigrid

[![CI](https://github.com/AzDevops143/GPU-problem1/actions/workflows/ci.yml/badge.svg)](https://github.com/AzDevops143/GPU-problem1/actions/workflows/ci.yml)
[![CUDA](https://img.shields.io/badge/CUDA-12.6%20%7C%2013.0-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/Benchmarked%20on-Tesla%20T4%20(sm__75)-76B900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Paper](https://img.shields.io/badge/Report-IEEE%20PDF-1F3B73.svg)](G25ait2026_IEEE_report_final.pdf)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter%20%2F%20Colab-2A9D8F.svg)](G25ait2026_final.ipynb)

A CUDA C++ study of the steady-state 2D heat equation ($\nabla^2 T = 0$) on an $N \times N$ Dirichlet plate. The project starts from textbook Jacobi iteration, moves to Red-Black SOR, and ends with a **Geometric Multigrid (GMG)** solver that converges in a constant number of cycles, runs as a single **CUDA Graph**, and detects convergence entirely on the device.

**Headline result (Tesla T4, FP64, $\varepsilon = 10^{-4}$, $N = 1025$):** GMG solves in **8.35 ms**, against 442.7 ms for optimally tuned RB-SOR (**53×**) and about 57 s for Jacobi (**~6,800×**), while landing **10,000× closer** to the exact discrete solution.

<p align="center">
  <img src="assets/fig_field.png" width="380" alt="Converged temperature field, N = 512">
  <img src="assets/fig_gmg_iters.png" width="420" alt="Iterations to converge versus grid size">
</p>

---

## Contents

- [Authors](#authors)
- [Key findings](#key-findings)
- [Problem formulation](#problem-formulation)
- [Solvers](#solvers)
- [Results](#results)
- [Quick start](#quick-start)
- [Command-line reference](#command-line-reference)
- [Output format](#output-format)
- [Repository layout](#repository-layout)
- [Reproducing figures and tables](#reproducing-figures-and-tables)
- [Continuous integration](#continuous-integration)
- [References](#references)

---

## Authors

GPU Programming Assignment, Problem 1, submitted to **Dr. Binod Kumar**.

| Name | Roll number |
|:---|:---|
| Peteti Charantej | G25AIT2026 |
| Aryaveer Rathi | G25AIT2021 |
| Sandeep Kumar Guntakal | G25AIT2036 |

---

## Key findings

1. **The algorithm matters far more than kernel tuning.**
   Moving from Jacobi (JOR) to RB-SOR with the optimal $\omega^*$ cuts iterations by **145×** at $N = 512$ (150,654 → 1,040). Multigrid then removes the dependence on $N$ altogether: **3 V(2,2)+FMG cycles at every size from 129 to 4097**.

2. **Convergence detection never leaves the GPU.**
   Each sweep computes $\max|T^{k+1} - T^k|$ with warp-shuffle reductions (`__shfl_down_sync`) and a block-level `atomicMax` on the IEEE-754 bit pattern of the non-negative value (`atomicMaxNonNeg`). Kernels skip their work once the flag is set, and the host polls a 4-byte flag only every $K$ iterations (default $K = 200$ for `heat2d`). The grid crosses PCIe exactly twice: once up, once down.

3. **Shared-memory tiling is slower than plain global memory on Turing.**
   Tiled kernels with halo staging run **2–16 % slower** across all sizes (global/shared time ratio 0.86–0.98). On the T4, L1 and shared memory are the same on-chip array, so L1 already captures the 5-point stencil reuse; explicit tiling only adds `__syncthreads()` barriers, halo loads and register pressure. The global kernels reach up to **99 GB/s** effective bandwidth.

4. **Small residual updates do not mean small error.**
   The stopping test $\|T^{k+1} - T^k\|_\infty < \varepsilon$ measures step size, not error. With spectral radius $\rho \approx 1 - \mathcal{O}(h^2)$ the true error can be up to $\frac{\rho}{1-\rho}$ times larger. At $N = 1024$, JOR stops **21 °C** away from the discrete solution; RB-SOR stops within $4.3 \times 10^{-3}$ °C; GMG within $4.5 \times 10^{-7}$ °C.

5. **FP64 is required for tight tolerances.**
   In float32, fine-grid SOR and GMG stall on a round-off plateau (V(2,2)+FMG needs 58 cycles at $N = 513$ and fails to reach $10^{-4}$ within 400 at $N \ge 1025$). FP64 converges cleanly in 3 cycles.

6. **CUDA Graphs help only where launch overhead dominates.**
   Capturing a whole V-cycle as one graph saves ~12 % at $N = 129$ and becomes negligible from $N = 1025$ upwards, where the cycle is bandwidth-bound.

---

## Problem formulation

Steady-state temperature on the unit square $\Omega = [0,1]^2$:

$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

Discretised on an $N \times N$ grid with $h = 1/(N-1)$, the 5-point stencil gives the Jacobi update

$$T_{i,j}^{k+1} = \tfrac{1}{4}\left(T_{i-1,j}^{k} + T_{i+1,j}^{k} + T_{i,j-1}^{k} + T_{i,j+1}^{k}\right)$$

**Boundary and initial conditions (defaults, all overridable on the command line):**

| Edge | Temperature |
|:---|:---:|
| Top | 100 °C |
| Left | 75 °C |
| Right | 50 °C |
| Bottom | 0 °C |
| Interior (initial guess) | 0 °C |

By symmetry, the centre of the plate is the average of the four edge temperatures, $(100 + 75 + 50 + 0)/4 = 56.25$ °C; the GPU solvers reproduce this value.

---

## Solvers

### 1. Jacobi over-relaxation (JOR) — `heat2d.cu`

- Ping-pong between two $N \times N$ buffers; $\omega = 1$ is plain Jacobi.
- $\rho_J = \cos(\pi h) \approx 1 - \pi^2 h^2 / 2$, so iterations grow as $\mathcal{O}(N^2)$.
- Over-relaxation does not help: JOR diverges for $\omega > 2/(1+\rho_J) \approx 1.01$ (confirmed at $\omega = 1.01$, $N = 256$).
- Two kernel variants: `jor_global` (direct global loads) and `jor_shared` (tile + halo in shared memory, 32×8 blocks).

### 2. Red-Black Gauss-Seidel and SOR (RB-GS, RB-SOR) — `heat2d.cu`

- Cells are coloured by the parity of $i + j$; each half-sweep reads only the other colour, so the update is in place on a single buffer.
- $\omega = 1$ gives RB-GS. By default `heat2d` uses Young's optimal factor:

$$\omega^* = \frac{2}{1 + \sin(\pi h)}, \qquad \rho_{\text{SOR}} = \omega^* - 1$$

- Iterations drop from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$. At $N = 256$, $\omega^* = 1.9757$ gives 531 iterations against 32,483 at $\omega = 1$; both $\omega = 1.95$ and $\omega = 1.98$ are measurably worse.

### 3. Geometric Multigrid (GMG) — `gmg.cu`

- Hierarchy $N_0 = N,\ N_{l+1} = (N_l - 1)/2 + 1$ down to $3 \times 3$. Use $N = 2^k + 1$ (129, 257, …, 4097) for a full hierarchy.
- **Smoother:** red-black Gauss-Seidel, $\nu_1 = 2$ pre- and $\nu_2 = 2$ post-sweeps by default.
- **Residual:** $r = f - A_h u$.
- **Restriction:** full weighting, $\frac{1}{16}\begin{bmatrix}1&2&1\\2&4&2\\1&2&1\end{bmatrix}$.
- **Prolongation:** bilinear interpolation, added as a correction.
- **Cycles:** V or W, optionally preceded by Full Multigrid (FMG) to build a good initial guess (on by default).
- **CUDA Graph:** an entire cycle (about 100 kernel launches) is captured once and replayed, removing CPU launch latency.
- **Device-side stop:** a `Ctrl` struct in device memory holds the stop flag; every kernel checks it first, so cycles queued past convergence cost almost nothing.

---

## Results

All measurements: **Google Colab, NVIDIA Tesla T4** (40 SMs, 16 GB GDDR6, 320 GB/s peak), CUDA 13.0, `nvcc -O3 -arch=sm_75`, FP64, $\varepsilon = 10^{-4}$. Raw data is in [`data/`](data) and [`gdata/`](gdata).

### Iterations and error versus the exact discrete solution

| $N$ | JOR iters | RB-GS iters | RB-SOR iters | JOR error (°C) | RB-SOR error (°C) |
|---:|---:|---:|---:|---:|---:|
| 64 | 5,655 | 3,107 | 144 | 0.08 | 8.6e-4 |
| 128 | 18,404 | 10,336 | 276 | 0.33 | 1.6e-3 |
| 256 | 55,830 | 32,483 | 531 | 1.32 | 2.7e-3 |
| 512 | 150,654 | 93,660 | 1,040 | 5.29 | 3.4e-3 |
| 1024 | 311,637 | 228,375 | 2,035 | 21.01 | 4.3e-3 |
| 2048 | — | — | 4,014 | — | 5.2e-3 |
| 4096 | — | — | 7,869 | — | 6.8e-3 |

Fitted growth: JOR $\sim N^{1.46}$, RB-GS $\sim N^{1.56}$, RB-SOR $\sim N^{0.96}$. JOR and RB-GS sit below the asymptotic $N^2$ because at large $N$ the loose stopping rule fires early — which is exactly why their error grows so fast.

### GMG convergence: cycles to $\varepsilon$ (constant in $N$)

| Variant | 129 | 513 | 1025 | 4097 | Convergence factor $\rho$ |
|:---|---:|---:|---:|---:|---:|
| V(1,1)+FMG | 4 | 4 | 4 | 4 | 0.060 |
| **V(2,2)+FMG** (default) | **3** | **3** | **3** | **3** | **0.036** |
| W(2,2)+FMG | 3 | 3 | 3 | 3 | 0.031 |
| V(2,2), no FMG | 6 | 6 | 6 | 6 | 0.043 |
| V(2,2)+FMG, float32 | 3 | 58 | >400 | >400 | — |

GMG error against the exact discrete solution stays between $3.1 \times 10^{-7}$ (N = 129) and $5.8 \times 10^{-7}$ °C (N = 4097).

### Time to solution: JOR / RB-GS / RB-SOR, global vs shared memory

| Solver | $N$ | Iters | Global (ms) | Shared (ms) | Global / Shared | GB/s |
|:---|---:|---:|---:|---:|---:|---:|
| JOR | 256 | 55,830 | 666.6 | 730.6 | 0.91× | 86 |
| JOR | 512 | 150,654 | 6,303.8 | 7,032.9 | 0.90× | 99 |
| JOR | 1024 | 311,637 | 56,922.0 | 63,586.0 | 0.90× | 91 |
| RB-GS | 512 | 93,660 | 4,345.6 | 4,839.0 | 0.90× | 90 |
| RB-GS | 1024 | 228,375 | 47,076.0 | 54,799.0 | 0.86× | 81 |
| RB-SOR | 512 | 1,040 | 45.0 | 50.3 | 0.89× | 96 |
| RB-SOR | 1024 | 2,035 | 412.1 | 478.2 | 0.86× | 83 |
| RB-SOR | 4096 | 7,869 | 25,383.0 | 29,435.0 | 0.86× | 83 |

Full table (all sizes): [`assets/tab_gpu.tex`](assets/tab_gpu.tex), raw CSV: [`data/gpu_bench.csv`](data/gpu_bench.csv).

### Time to solution: RB-SOR vs GMG V(2,2)+FMG with CUDA Graph

| $N$ | RB-SOR iters | RB-SOR (ms) | GMG cycles | Levels | GMG (ms) | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 129 | 278 | 3.2 | 3 | 7 | 1.54 | 2× |
| 257 | 533 | 8.4 | 3 | 8 | 1.94 | 4× |
| 513 | 1,042 | 52.5 | 3 | 9 | 2.99 | 18× |
| 1025 | 2,033 | 442.7 | 3 | 10 | **8.35** | **53×** |
| 2049 | 4,015 | 3,351.8 | 3 | 11 | 30.14 | 111× |
| 4097 | 7,881 | 25,766.5 | 3 | 12 | 120.95 | **213×** |

<p align="center">
  <img src="assets/fig_gmg_time.png" width="800" alt="Time to solution and CUDA Graph effect">
</p>

### Asymptotic summary

| | JOR ($\omega = 1$) | RB-SOR ($\omega^*$) | GMG V(2,2)+FMG |
|:---|:---:|:---:|:---:|
| Iterations to $\varepsilon$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(N)$ | $\mathcal{O}(1)$ |
| Work per iteration | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^2)$ (≈ 10 work units) |
| Total work | $\mathcal{O}(N^4)$ | $\mathcal{O}(N^3)$ | $\mathcal{O}(N^2)$ — optimal |
| Memory | $2N^2$ | $N^2$ | ≈ $4N^2$ |
| Tuning parameter | none | $\omega^*$ | none |
| Measured time, $N \approx 1024$ | ~57 s | 412–443 ms | **8.35 ms** |
| Error at $\varepsilon = 10^{-4}$, $N \approx 1024$ | 21.01 °C | 4.3e-3 °C | **4.5e-7 °C** |

### Host polling interval

Polling the convergence flag every iteration costs about 27 % more than the best interval; the curve is flat between $K = 100$ and $K = 200$ ([`data/gpu_batch.csv`](data/gpu_batch.csv), [`assets/fig_batch.png`](assets/fig_batch.png)).

---

## Quick start

Requirements: an NVIDIA GPU (compute capability 7.5 or newer) and the CUDA Toolkit 12.x with `nvcc`, **or** Docker with the NVIDIA Container Toolkit.

### Option 1 — Build locally with `make`

```bash
git clone https://github.com/AzDevops143/GPU-problem1.git
cd GPU-problem1

make              # builds ./heat2d and ./gmg for sm_75, sm_80, sm_86, sm_89, sm_90 (+ PTX)
make run-heat2d   # ./heat2d -n 256 -e 1e-4 -m all -v both -p double
make run-gmg      # ./gmg   -n 513 -e 1e-4 -c V -p double
make clean
```

Building for a single GPU is faster, e.g. on a T4:

```bash
nvcc -O3 -std=c++17 -arch=sm_75 heat2d.cu -o heat2d
nvcc -O3 -std=c++17 -arch=sm_75 gmg.cu    -o gmg
```

### Option 2 — Docker

The image (`nvidia/cuda:12.6.3-devel-ubuntu22.04`) compiles both solvers and includes JupyterLab with NumPy, pandas and Matplotlib.

```bash
docker build -t gpu-heat-diffusion:latest .               # or: make docker-build

docker run --gpus all --rm -it gpu-heat-diffusion:latest  # runs ./gmg -n 513 -e 1e-4

docker run --gpus all --rm -it gpu-heat-diffusion:latest ./heat2d -n 512 --verify

docker run --gpus all --rm -it -p 8888:8888 gpu-heat-diffusion:latest \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root   # or: make docker-jupyter
```

### Option 3 — Google Colab

1. Open [`G25ait2026_final.ipynb`](G25ait2026_final.ipynb) in Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. The notebook writes and compiles both `.cu` files, runs every experiment, saves CSVs to `data/` and `gdata/`, and regenerates all figures and tables in `assets/`.

### Correctness check

Both programs have a `--verify` flag that runs a sequential CPU implementation of the same algorithm and compares the final grids. It prints `PASS` when the maximum absolute difference is below $10\varepsilon$ (or $\max(0.05, 10\varepsilon)$ in float32); `gmg` additionally requires the CPU and GPU cycle counts to agree within one.

```bash
./heat2d -n 37 -e 1e-4 --verify     # odd size, both methods, both kernel variants
./heat2d -n 128 -p float --verify   # single precision
./gmg -n 65 -e 1e-4 --verify
```

---

## Command-line reference

### `heat2d` — JOR and Red-Black SOR

```
./heat2d [-n N] [-e eps] [-m jor|rbsor|all] [-v global|shared|both]
         [-p double|float] [-w omega] [--maxit M] [--batch K]
         [-i init.bin] [-o out.bin] [--hist hist.csv]
         [--top T --bottom T --left T --right T --init T] [--verify] [-q]
```

| Flag | Default | Meaning |
|:---|:---|:---|
| `-n` | `512` | Grid size $N$ (≥ 3) |
| `-e` | `1e-4` | Stop when $\max\lvert T^{k+1}-T^k \rvert < \varepsilon$ |
| `-m` | `all` | `jor`, `rbsor`, or both |
| `-v` | `both` | Kernel variant: `global`, `shared`, or both |
| `-p` | `double` | Precision: `double` or `float` |
| `-w` | auto | Relaxation factor; auto = 1 for JOR, $\omega^*$ for RB-SOR (use `-w 1` for RB-GS) |
| `--maxit` | `1000000` | Iteration cap |
| `--batch` | `200` | Host polls the convergence flag every $K$ iterations |
| `-i` / `-o` | — | Read initial grid / write result as raw $N \times N$ float64, row-major. Edges of `init.bin` are the boundary conditions |
| `--hist` | — | Write per-iteration `iter,maxdiff` to `<method>_<variant>_<file>` |
| `--top` … `--init` | 100 / 0 / 75 / 50 / 0 | Boundary and initial temperatures (°C) |
| `--verify` | off | Compare with a CPU reference |
| `-q` | off | Suppress the `#` diagnostic lines |

### `gmg` — Geometric Multigrid

```
./gmg [-n N] [-e eps] [-c V|W] [--nu1 k] [--nu2 k] [-p double|float]
      [--levels L] [--coarse-sweeps k] [--maxcycles M] [--batch K]
      [--no-fmg] [--no-graph] [--crit diff|res] [-s Q]
      [-i init.bin] [-o out.bin] [--hist hist.csv]
      [--top T --bottom T --left T --right T --init T] [--verify] [-q]
```

| Flag | Default | Meaning |
|:---|:---|:---|
| `-n` | `513` | Grid size; use $2^k + 1$ for a full hierarchy |
| `-e` | `1e-4` | Convergence tolerance |
| `-c` | `V` | Cycle type: `V` or `W` |
| `--nu1`, `--nu2` | `2`, `2` | Pre- and post-smoothing sweeps |
| `-p` | `double` | Precision |
| `--levels` | all | Cap on the number of grid levels |
| `--coarse-sweeps` | auto | Smoothing sweeps on the coarsest grid |
| `--maxcycles` | `200` | Cycle cap |
| `--batch` | `2` | Cycles between host polls |
| `--no-fmg` | — | Start from the initial guess instead of Full Multigrid |
| `--no-graph` | — | Launch kernels individually instead of via a CUDA Graph |
| `--crit` | `diff` | Stop on `diff` ($\max\lvert u - u_{prev}\rvert$) or `res` (max residual) |
| `-s` | `0` | Uniform heat source $Q$ (solves $-\nabla^2 T = Q$) |
| `-i`, `-o`, `--hist`, BC flags, `--verify`, `-q` | | As for `heat2d`; `--hist` writes `cycle,maxdiff,maxres` |

`gmg` exits with status `0` when converged, `2` if it hit `--maxcycles` or diverged.

If no CUDA device is present (for example on a CI runner), both binaries print a notice and exit `0` without running.

---

## Output format

Every run prints a machine-readable CSV header line and one row per configuration, both prefixed with `RESULT,` so they can be grepped straight into pandas:

```text
RESULT,method,variant,precision,N,omega,eps,iters,converged,final_diff,kernel_ms,total_ms,MLUPS,GBps          # heat2d
RESULT,method,cycle,precision,N,levels,coarsest,nu1,nu2,fmg,graph,cycles,converged,final_diff,final_res,
       conv_factor,fmg_ms,solve_ms,ms_per_cycle,total_ms,MLUPS_cycle                                         # gmg
```

`--verify` adds a `VERIFY,…,PASS|FAIL` line. Lines starting with `#` are human-readable diagnostics (device, hierarchy, per-cycle residuals, sample temperatures).

---

## Repository layout

```text
.
├── heat2d.cu                         # JOR and Red-Black SOR solvers (global + shared-memory kernels)
├── gmg.cu                            # Geometric Multigrid solver with CUDA Graphs and FMG
├── G25ait2026_final.ipynb            # Full experiment notebook (Colab, Tesla T4)
├── G25ait2026_IEEE_report_final.pdf  # IEEE-format report
├── make_assets.py                    # Figures, LaTeX tables and numbers for JOR / RB-SOR (reads data/)
├── make_gmg_assets.py                # Figures, LaTeX tables and numbers for GMG (reads gdata/)
├── data/                             # Measured CSVs: iterations, ω sweep, benchmarks, polling interval
├── gdata/                            # Measured CSVs: GMG cycles, timings, accuracy, ν and level studies
├── assets/                           # 13 PNG figures, LaTeX tables, summary numbers (numbers.tex/.json)
├── Makefile                          # Build, run and Docker targets
├── Dockerfile                        # CUDA 12.6 image with both solvers and JupyterLab
├── .dockerignore
└── .github/workflows/ci.yml          # Three-stage CI pipeline
```

---

## Reproducing figures and tables

The committed `assets/` were produced by the notebook from the committed measurements. To rebuild the JOR / RB-SOR assets from `data/`:

```bash
pip install numpy pandas matplotlib
python3 make_assets.py --data data --out assets
```

`make_gmg_assets.py --data gdata --out assets` additionally needs the convergence-history files (`hist_gmg_d.csv`, `hist_gmg_f.csv`, `hist_sor_1025.csv`) that the notebook generates with `--hist`; these are not committed, so the simplest way to rebuild the GMG figures is to run the notebook end to end on a GPU.

---

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and pull request to `main`:

```mermaid
graph LR
    A["1. Workspace integrity"] --> B["2. Docker build and CUDA compile"]
    B --> C["3. Package and upload artifacts"]
    A --> C
```

1. **Workspace integrity** — parses the notebook JSON and checks that all required source, build and report files exist.
2. **Docker build and CUDA compile** — builds the image (compiling for `sm_75`–`sm_90` plus PTX), runs both binaries inside it, and extracts the compiled executables. GitHub-hosted runners have no GPU, so this confirms that the code compiles and starts, not the numerical results.
3. **Package and upload** — bundles the compiled binaries and sources, `data/` and `gdata/`, the figures, the LaTeX tables, an HTML export of the notebook and the PDF report into the **`gpu-heat-diffusion-complete-artifacts`** artifact (kept 30 days).

To download it: open the [Actions tab](https://github.com/AzDevops143/GPU-problem1/actions), select the latest successful run, and scroll to **Artifacts**.

---

## References

1. D. M. Young, *Iterative Solution of Large Linear Systems*. Academic Press, 1971.
2. W. L. Briggs, V. E. Henson, and S. F. McCormick, *A Multigrid Tutorial*, 2nd ed. SIAM, 2000.
3. U. Trottenberg, C. W. Oosterlee, and A. Schüller, *Multigrid*. Academic Press, 2001.
4. M. Harris, "Optimizing Parallel Reduction in CUDA," NVIDIA Developer Technology, 2007.
5. NVIDIA Corporation, *CUDA C++ Programming Guide* (CUDA Graphs, warp shuffle intrinsics).
