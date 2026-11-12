import math
import mpmath

mpmath.mp.dps = 50

def relative_error(approx, exact):
    approx = float(approx)
    exact = float(exact)
    dx = abs(exact - approx)
    return dx / abs(exact)

print("Таблица относительных ошибок:")

print(f"{'x':<15} | {'Плохой способ':<20} | {'Хороший способ':<20}")
print("-" * 63)

arr_x = [10**10, 10**12, 10**14]

for x in arr_x:
    bad_res = math.sqrt(x + 1) - math.sqrt(x)
    good_res = 1 / (math.sqrt(x + 1) + math.sqrt(x))
    
    x_mp = mpmath.mpf(x)
    exact_res = mpmath.sqrt(x_mp + 1) - mpmath.sqrt(x_mp)
    
    err_bad = relative_error(bad_res, exact_res)
    err_good = relative_error(good_res, exact_res)
    
    
    print(f"{x:<15} | {err_bad:<20.2e} | {err_good:<20.2e}")

print()


print("ВЫВОД")
print("Как видно из таблицы, при росте x расхождение 'плохого' способа стремительно увеличивается.")
print("Это происходит из-за того, что корни становятся почти одинаковыми, и при их вычитании (катастрофическое сокращение) мы теряем точность.")
print("В 'хорошем' способе мы заменили вычитание на сложение в знаменателе, поэтому его относительная ошибка всегда остается крошечной (на уровне нуля).")