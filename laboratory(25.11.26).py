import math

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
    """Сколько делений нужно, чтобы длина стала меньше eps."""
    return math.ceil(math.log2(length / eps))


def f_poly(x):
    return x**3 - 6 * x**2 + 11 * x - 6


def f_param(x, k):
    return math.cos(x) - k * x


# ── Задача 1 ──────────────────────────────────────────────

print("Задача 1. Корни f(x) = x^3 - 6x^2 + 11x - 6 на [0, 4]")
print("Аналитически: (x - 1)(x - 2)(x - 3) = 0, корни 1, 2 и 3.")
print()
print("Значения на сетке, по ним видны смены знака:")
print(f"{'x':>6} | {'f(x)':>10}")
print("-" * 20)
for x in (0.6, 1.3, 1.2, 2.7, 2.2, 4.0):
    print(f"{x:6.1f} | {f_poly(x):10.4f}")

# Интервалы разной длины. Середина не совпадает с корнем,
# иначе метод остановится на первом шаге из-за f(c) = 0.
brackets = ((0.6, 1.3), (1.2, 2.7), (2.2, 4.0))
exact_roots = (1.0, 2.0, 3.0)
eps = 1e-9

xs = np.linspace(0, 4, 400)
plt.figure(figsize=(8, 4.5))
plt.axhline(0, color="black", linewidth=0.8)
plt.plot(xs, f_poly(xs), label="f(x)")
for a, b in brackets:
    plt.axvspan(a, b, alpha=0.15)
plt.scatter(exact_roots, [0, 0, 0], color="red", zorder=3, label="корни 1, 2, 3")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) = x³ − 6x² + 11x − 6 на [0, 4]")
plt.legend()
plt.tight_layout()
graph1 = "laboratory_25_11_26_task1.png"
plt.savefig(graph1, dpi=120)
plt.close()
print(f"\nГрафик сохранён: {graph1}")

task1 = []
for (a, b), exact in zip(brackets, exact_roots):
    root, journal, status = bisection(f_poly, a, b, eps=eps)
    print_journal(journal, f"корень около {exact:g}, интервал [{a}, {b}]")
    abs_err = abs(root - exact)
    rel_err = abs_err / abs(exact)
    lengths_ok = all(
        math.isclose(row["length"], (b - a) / 2**i, rel_tol=0, abs_tol=1e-9)
        for i, row in enumerate(journal)
    )
    task1.append(
        {
            "exact": exact,
            "root": root,
            "n": len(journal),
            "abs_err": abs_err,
            "rel_err": rel_err,
            "status": status,
            "length0": b - a,
            "theory": theory_steps(b - a, eps),
            "halves": lengths_ok,
            "f_root": f_poly(root),
        }
    )
    print(
        f"корень: {root:.12f}, f(root) = {f_poly(root):.3e}, "
        f"статус: {status}, итераций: {len(journal)}"
    )
    print(f"длина интервала делится пополам: {'да' if lengths_ok else 'нет'}")

print("\nТаблица. корень → итерации → Δ → δ")
print(
    f"{'x*':>4} | {'x числ.':>14} | {'n':>3} | {'Δ0':>6} | "
    f"{'log2(Δ0/ε)':>11} | {'Δ':>12} | {'δ':>12} | статус"
)
print("-" * 86)
for row in task1:
    print(
        f"{row['exact']:4.0f} | {row['root']:14.10f} | {row['n']:3d} | "
        f"{row['length0']:6.2f} | {row['theory']:11d} | "
        f"{row['abs_err']:12.3e} | {row['rel_err']:12.3e} | {row['status']}"
    )

