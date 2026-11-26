import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bisection(f, a, b, eps=1e-9, max_iter=200):
    """Метод половинного деления с журналом итераций.

    Возвращает: (корень, журнал, статус)
    статус = 'ok' — достигнута точность по длине интервала
    статус = 'f_zero' — f(c) == 0 (точное попадание)
    статус = 'max_iter' — исчерпан лимит итераций
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(
            f"Нет смены знака на [{a}, {b}]: f(a)={fa}, f(b)={fb}"
        )

    journal = []

    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        length = b - a
        journal.append(
            {
                "iter": k,
                "a": a,
                "b": b,
                "c": c,
                "f(c)": fc,
                "length": length,
            }
        )

        if fc == 0:
            return c, journal, "f_zero"
        if length < eps:
            return c, journal, "ok"

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, journal, "max_iter"


def print_journal(journal, title=""):
    if title:
        print(f"\n=== {title} ===")
    print(
        f"{'k':>3} | {'a':>12} | {'b':>12} | {'c':>12} | "
        f"{'f(c)':>14} | {'b-a':>10}"
    )
    print("-" * 80)
    for row in journal:
        print(
            f"{row['iter']:>3} | {row['a']:>12.6f} | {row['b']:>12.6f} | "
            f"{row['c']:>12.6f} | {row['f(c)']:>14.3e} | "
            f"{row['length']:>10.2e}"
        )


def theory_steps(length, eps):
    return math.ceil(math.log2(length / eps))


def f_param(x, k):
    return math.cos(x) - k * x


def solve_k(k, a=0.0, b=2.0, tol=1e-9):
    return bisection(lambda x: f_param(x, k), a, b, eps=tol)


print("Задача 2. f(x) = cos(x) - kx = 0")

root_k, journal_k, status_k = solve_k(0.5)
print(
    f"k = 0.5, интервал [0, 2]: f(0) = {f_param(0, 0.5):.4f}, "
    f"f(2) = {f_param(2, 0.5):.4f}, смена знака есть."
)
print_journal(journal_k, "k = 0.5, [0, 2], ε = 1e-9")
print(
    f"корень: {root_k:.12f}, f(root) = {f_param(root_k, 0.5):.3e}, "
    f"статус: {status_k}, итераций: {len(journal_k)}"
)

print("\nТаблица. k → корень → число итераций → |f(корень)|")
print(f"{'k':>4} | {'корень':>14} | {'n':>3} | {'|f|':>12} | статус")
print("-" * 48)
for k in (0.5, 1, 2, 5):
    root, journal, status = solve_k(k)
    print(
        f"{k:4.1f} | {root:14.10f} | {len(journal):3d} | "
        f"{abs(f_param(root, k)):12.3e} | {status}"
    )

k_grid = np.arange(0.1, 10.0 + 1e-9, 0.1)
root_grid = np.array([solve_k(float(k))[0] for k in k_grid])

plt.figure(figsize=(8, 4.5))
plt.plot(k_grid, root_grid)
plt.axhline(math.pi / 2, color="gray", linestyle="--", linewidth=0.8, label="π/2")
plt.xlabel("k")
plt.ylabel("корень")
plt.title("Корень cos(x) − kx = 0 в зависимости от k")
plt.legend()
plt.tight_layout()
graph_k = Path(__file__).with_name("laboratory_25_11_26_root_vs_k.png")
plt.savefig(graph_k, dpi=120)
plt.close()
print(f"\nГрафик «корень от k» сохранён: {graph_k}")
print(f"при k = 0.1 корень = {root_grid[0]:.6f}")
print(f"при k = 10  корень = {root_grid[-1]:.6f}")
print(f"π/2 = {math.pi / 2:.6f}")

x_star, _, _ = solve_k(0.5, tol=1e-15)
print("\nТаблица. ε → итерации → |x − x*|   (k = 0.5, x* при ε = 1e-15)")
print(f"x* = {x_star:.15f}")
print(f"{'ε':>8} | {'n':>3} | {'log2(Δ0/ε)':>11} | {'|x-x*|':>12}")
print("-" * 44)
eps_rows = []
for tol in (1e-3, 1e-6, 1e-9, 1e-12):
    root, journal, _ = solve_k(0.5, tol=tol)
    err = abs(root - x_star)
    estimate = theory_steps(2.0, tol)
    eps_rows.append((tol, len(journal), estimate, err))
    print(f"{tol:8.0e} | {len(journal):3d} | {estimate:11d} | {err:12.3e}")

plt.figure(figsize=(8, 4.5))
plt.plot(
    [row[0] for row in eps_rows],
    [row[1] for row in eps_rows],
    marker="o",
    label="факт, n",
)
plt.plot(
    [row[0] for row in eps_rows],
    [row[2] for row in eps_rows],
    marker="s",
    label="оценка log2(Δ0/ε)",
)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel("ε")
plt.ylabel("число итераций")
plt.title("Число итераций при разной точности, k = 0.5")
plt.legend()
plt.tight_layout()
graph_eps = Path(__file__).with_name("laboratory_25_11_26_iters_vs_eps.png")
plt.savefig(graph_eps, dpi=120)
plt.close()
print(f"График «итерации от ε» сохранён: {graph_eps}")

print("\nk = 0: уравнение cos(x) = 0 на [0, 2]")
print(f"f(0) = {f_param(0, 0):.4f}, f(2) = {f_param(2, 0):.4f}")
root0, journal0, status0 = solve_k(0.0)
print(
    f"корень: {root0:.12f}, π/2 = {math.pi / 2:.12f}, "
    f"итераций: {len(journal0)}, статус: {status0}"
)

print("\nВЫВОД")
print(
    "1. При k → 0 корень стремится к π/2, потому что уравнение становится "
    "cos(x) = 0. При росте k прямая kx становится круче и корень сдвигается к нулю."
)
print(
    "2. Число итераций почти не зависит от k, если стартовый интервал один и тот же. "
    "Оно близко к log2(Δ0/ε). В журнале на одну строку больше оценки: остановка "
    "проверяется в начале шага, когда длина уже меньше ε."
)
print(
    "3. При k = 0 бисекция на [0, 2] работает: f(0) = 1 > 0, f(2) = cos(2) < 0, "
    "корень π/2 лежит внутри. Менять интервал не нужно. Метод отказал бы на отрезке "
    "без смены знака, например на [0, 1]."
)
