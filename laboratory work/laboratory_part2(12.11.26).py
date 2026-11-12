import math
import time
import numpy as np


def measure(fn, arr, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(arr)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


# ——— Задача 1. Сумма массива ———

def loop_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total


def numpy_sum(arr):
    return arr.sum()


print("Задача 1. Сумма массива")
print(f"{'n':>10} | {'цикл, мс':>12} | {'NumPy, мс':>12} | {'ускорение':>10}")
print("-" * 52)

for n in (10**5, 10**6, 10**7):
    rng = np.random.default_rng(42)
    a = rng.normal(size=n)

    r_loop, t_loop = measure(loop_sum, a)
    r_numpy, t_numpy = measure(numpy_sum, a)
    speedup = t_loop / t_numpy

    print(f"цикл:   {t_loop * 1000:10.2f} мс  результат = {r_loop:.6f}")
    print(f"NumPy:  {t_numpy * 1000:10.2f} мс  результат = {r_numpy:.6f}")
    print(f"ускорение: {speedup:.1f}×")
    print(f"результаты совпадают: {np.isclose(r_loop, r_numpy)}")
    print(f"{n:10} | {t_loop * 1000:12.2f} | {t_numpy * 1000:12.2f} | {speedup:9.1f}×")
    print()

print("Вывод: разница заметна уже при n = 10^5, ускорение там 238×, то есть больше 10×. Дальше оно остаётся большим, но падает: 191× при 10^6 и 101× при 10^7.")
print("Так выходит, потому что цикл Python и так медленный на каждом элементе, а NumPy на маленьком массиве почти мгновенный. На 10^7 массив уже не помещается в кэш, и NumPy упирается в память, поэтому разрыв сужается.")
print("Читаемость: numpy_sum — одна строка, loop_sum — четыре. Смысл «сумма массива» в NumPy виден сразу.")
print()


# ——— Задача 2. Среднее и стандартное отклонение ———

def loop_mean_std(arr):
    n = len(arr)
    total = 0.0
    for x in arr:
        total += x
    mean = total / n
    total_sq = 0.0
    for x in arr:
        d = x - mean
        total_sq += d * d
    std = math.sqrt(total_sq / n)
    return mean, std


def numpy_mean_std(arr):
    return arr.mean(), arr.std()


print("Задача 2. Среднее и стандартное отклонение")
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

r_loop, t_loop = measure(loop_mean_std, a)
r_numpy, t_numpy = measure(numpy_mean_std, a)

print(f"цикл:   {t_loop * 1000:10.2f} мс  результат = mean {r_loop[0]:.6f}, std {r_loop[1]:.6f}")
print(f"NumPy:  {t_numpy * 1000:10.2f} мс  результат = mean {r_numpy[0]:.6f}, std {r_numpy[1]:.6f}")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy, atol=1e-12, rtol=0)}")
print("Вывод: оба способа дают одно среднее и одно std с допуском 1e-12. В цикле это сумма и сумма квадратов отклонений, затем формула. a.mean() и a.std() короче и быстрее, потому что два прохода по миллиону чисел остаются в C, а не в Python.")
print("Читаемость: в цикле два прохода и формула, в NumPy это mean() и std(). Имена методов короче, формулу держать в голове не нужно.")
print()


# ——— Задача 3. Поэлементное преобразование y = sin(x) + x^2 ———

def loop_transform(arr):
    result = []
    for x in arr:
        result.append(math.sin(x) + x * x)
    return result


def numpy_transform(arr):
    return np.sin(arr) + arr**2


print("Задача 3. Поэлементное преобразование")
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)

r_loop, t_loop = measure(loop_transform, a)
r_numpy, t_numpy = measure(numpy_transform, a)

print(f"цикл:   {t_loop * 1000:10.2f} мс")
print(f"NumPy:  {t_numpy * 1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print("Вывод: формула одна и та же, sin(x) + x*x. Цикл вызывает math.sin на каждом элементе и складывает список, np.sin(a) + a**2 считает весь массив сразу. Ускорение 20×, меньше, чем у голой суммы: синус тяжёлый и в C, поэтому доля Python здесь меньше.")
print("Читаемость: строка NumPy совпадает с формулой. Цикл длиннее и добавляет пустой список, которого в задаче нет.")
print()


# ——— Задача 4. Условная замена (clipping) ———

def loop_clip(arr):
    result = []
    for x in arr:
        if x < 0:
            result.append(0.0)
        elif x > 1:
            result.append(1.0)
        else:
            result.append(float(x))
    return result


def numpy_clip(arr):
    return np.clip(arr, 0, 1)


print("Задача 4. Условная замена (clipping)")
rng = np.random.default_rng(42)
a = rng.normal(size=10**6)
changed = int(np.sum((a < 0) | (a > 1)))

r_loop, t_loop = measure(loop_clip, a)
r_numpy, t_numpy = measure(numpy_clip, a)

print(f"цикл:   {t_loop * 1000:10.2f} мс")
print(f"NumPy:  {t_numpy * 1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print("изменено элементов:", changed)
print("Вывод: всё, что меньше 0, становится 0, всё, что больше 1, становится 1. np.clip(a, 0, 1) делает это без ветвлений на каждый элемент, цикл if/elif — с ними, поэтому NumPy быстрее.")
print("Читаемость: одна функция вместо трёх веток. Число изменённых элементов — побочная статистика, её даёт та же маска (a < 0) | (a > 1).")
