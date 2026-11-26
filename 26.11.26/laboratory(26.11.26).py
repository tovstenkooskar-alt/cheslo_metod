import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def newton(f, df, x0, eps=1e-12, max_iter=100):
    """Метод Ньютона (касательных).

    Возвращает: (корень, журнал, статус)
    """
    journal = []
    x = x0

    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return x, journal, "zero_derivative"

        x_new = x - fx / dfx
        journal.append(
            {
                "iter": k,
                "x": x,
                "f(x)": fx,
                "f'(x)": dfx,
                "x_new": x_new,
                "step": abs(x_new - x),
            }
        )

        if abs(x_new - x) < eps or abs(fx) < eps:
            return x_new, journal, "ok"
        x = x_new

    return x, journal, "max_iter"


def secant(f, x0, x1, eps=1e-12, max_iter=100):
    """Метод секущих.

    Возвращает: (корень, журнал, статус)
    """
    journal = []
    x_prev, x_curr = x0, x1
    f_prev, f_curr = f(x_prev), f(x_curr)

    for k in range(1, max_iter + 1):
        if f_curr - f_prev == 0:
            return x_curr, journal, "zero_denominator"

        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        journal.append(
            {
                "iter": k,
                "x_prev": x_prev,
                "x_curr": x_curr,
                "f_curr": f_curr,
                "x_new": x_new,
                "step": abs(x_new - x_curr),
            }
        )

        if abs(x_new - x_curr) < eps or abs(f_curr) < eps:
            return x_new, journal, "ok"

        x_prev, f_prev = x_curr, f_curr
        x_curr = x_new
        f_curr = f(x_curr)

    return x_curr, journal, "max_iter"


def bisection(f, a, b, eps=1e-12, max_iter=200):
    """Метод половинного деления с журналом итераций."""
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

        if fc == 0 or length < eps:
            return c, journal, "ok"

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, journal, "max_iter"


def print_journal_newton(journal, title=""):
    if title:
        print(f"\n=== {title} ===")
    print(
        f"{'k':>3} | {'x':>14} | {'f(x)':>14} | {'df(x)':>14} | {'шаг':>10}"
    )
    print("-" * 70)
    for row in journal:
        print(
            f"{row['iter']:>3} | {row['x']:>14.10f} | {row['f(x)']:>14.3e} | "
            f"{row['f\'(x)']:>14.3e} | {row['step']:>10.2e}"
        )


def print_journal_secant(journal, title=""):
    if title:
        print(f"\n=== {title} ===")
    print(
        f"{'k':>3} | {'x_prev':>14} | {'x_curr':>14} | {'f_curr':>14} | {'шаг':>10}"
    )
    print("-" * 70)
    for row in journal:
        print(
            f"{row['iter']:>3} | {row['x_prev']:>14.10f} | {row['x_curr']:>14.10f} | "
            f"{row['f_curr']:>14.3e} | {row['step']:>10.2e}"
        )


def print_journal_bisection(journal, title=""):
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


def f(x):
    return x**3 - 2 * x + 2


def df(x):
    return 3 * x**2 - 2


print("Задача 1. f(x) = x^3 - 2x + 2")
print("Один действительный корень около -1.769.")

xs = np.linspace(-3, 3, 400)
plt.figure(figsize=(8, 4.5))
plt.axhline(0, color="black", linewidth=0.8)
plt.plot(xs, f(xs), label="f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) = x³ − 2x + 2 на [−3, 3]")
plt.legend()
plt.tight_layout()
graph = Path(__file__).with_name("laboratory_26_11_26_task1.png")
plt.savefig(graph, dpi=120)
plt.close()
print(f"График сохранён: {graph}")
print(f"f(-2) = {f(-2):.4f}, f(-1) = {f(-1):.4f}, смена знака на [-2, -1].")
print("На [-3, 3] ось X пересекается один раз.")

root_bad, journal_bad, status_bad = newton(f, df, x0=0.0)
print_journal_newton(
    journal_bad[:8],
    "Ньютон, x0 = 0. Первые 8 строк: дальше цикл 0 и 1",
)
print(
    f"статус: {status_bad}, итераций до остановки: {len(journal_bad)}. "
    "Точки ходят 0 → 1 → 0 → 1."
)

root_n, journal_n, status_n = newton(f, df, x0=-2.0)
print_journal_newton(journal_n, "Ньютон, x0 = -2")
print(
    f"корень: {root_n:.12f}, f(root) = {f(root_n):.3e}, "
    f"статус: {status_n}, итераций: {len(journal_n)}"
)

root_s, journal_s, status_s = secant(f, x0=-2.0, x1=-1.0)
print_journal_secant(journal_s, "Секущие, x0 = -2, x1 = -1")
print(
    f"корень: {root_s:.12f}, f(root) = {f(root_s):.3e}, "
    f"статус: {status_s}, итераций: {len(journal_s)}"
)

root_b, journal_b, status_b = bisection(f, -2.0, -1.0)
print_journal_bisection(journal_b, "Бисекция, [-2, -1]")
print(
    f"корень: {root_b:.12f}, f(root) = {f(root_b):.3e}, "
    f"статус: {status_b}, итераций: {len(journal_b)}"
)

print("\nТаблица. метод → старт → корень → итераций")
print(f"{'метод':<12} | {'старт':<16} | {'корень':>14} | {'n':>4} | статус")
print("-" * 62)
rows = (
    ("Ньютон", "x0 = 0", root_bad, len(journal_bad), status_bad),
    ("Ньютон", "x0 = -2", root_n, len(journal_n), status_n),
    ("Секущие", "x0=-2, x1=-1", root_s, len(journal_s), status_s),
    ("Бисекция", "[-2, -1]", root_b, len(journal_b), status_b),
)
for name, start, root, n, status in rows:
    root_text = f"{root:14.10f}" if status == "ok" else "   не сошёлся"
    print(f"{name:<12} | {start:<16} | {root_text} | {n:4d} | {status}")

print("\nВЫВОД")
print(
    "1. На [-3, 3] график пересекает ось X один раз, корень около -1.77. "
    "Смена знака есть на [-2, -1]: f(-2) < 0, f(-1) > 0."
)
print(
    "2. Ньютон с x0 = 0 не сходится. f(0) = 2, f'(0) = -2, следующая точка равна 1. "
    "f(1) = 1, f'(1) = 1, следующая точка снова 0. Метод ходит по кругу, "
    "хотя производная не ноль. Старт далеко от корня, между локальным максимумом "
    "и минимумом."
)
print(
    "3. С x0 = -2 Ньютон доходит до корня за несколько шагов: старт уже рядом. "
    "Секущие с точек -2 и -1 тоже сходятся и делают немного больше итераций. "
    "Бисекция на [-2, -1] сходится гарантированно, но шагов около log2(1/ε), "
    "то есть заметно больше."
)
print(
    "4. Плохое начальное приближение ломает Ньютона, даже если корень один. "
    "Бисекция от этого не зависит: ей нужен только отрезок со сменой знака."
)
