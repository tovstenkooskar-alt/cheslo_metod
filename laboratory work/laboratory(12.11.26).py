import numpy as np

print("1. Массив формы (6, 4)")
a = np.arange(24).reshape(6, 4)
print(a)
print("shape:", a.shape)
print("ndim:", a.ndim)
print("size:", a.size)
print("dtype:", a.dtype)
print()

print("2. Перестройка формы")
flat = a.reshape(24).copy()
b = a.reshape(4, 6)
c = b.reshape(2, 3, 4)
d = c.reshape(24)
print("Форма (4, 6):", b.shape)
print("Форма (2, 3, 4):", c.shape)
print("Форма обратно (24,):", d.shape)
print("Значения не изменились:", np.array_equal(d, flat))
print()

print("3. Столбец [:, 2]: view или копия")
col = a[:, 2]
print("Столбец:", col)
was = a[0, 2]
col[0] = 999
print("После замены первого элемента среза a[0, 2] =", a[0, 2])
print("Оригинал изменился, значит столбец.")
a[0, 2] = was
print("Значение вернул обратно, дальше считаю исходный массив.")
print()

print("4. Элементы больше 10")
picked = a[a > 10]
print(picked)
print("Среднее:", picked.mean())
print()

print("5. Вычитание среднего по строкам")
centered = a - a.mean(axis=1, keepdims=True)
row_sum = centered.sum(axis=1)
print("Форма результата:", centered.shape)
print("Форма та же, что у исходного массива:", centered.shape == a.shape)
print("Сумма по строкам:", row_sum)
print("Сумма по строкам равна нулю:", np.allclose(row_sum, 0))
print()

print("ВЫВОД")
print("1. Из 24 чисел получается массив 6 на 4. У него 2 оси, 24 элемента, тип целый.")
print("2. Смена формы не трогает сами числа: после (4, 6), (2, 3, 4) и обратно в (24,) массив совпадает с исходным.")
print("3. Столбец [:, 2] смотрит в тот же массив. Если поменять элемент среза, меняется и оригинал.")
print("4. Маска a > 10 оставляет числа с 11 по 23. Их среднее равно 17.")
print("5. Если из каждой строки вычесть её среднее, форма остаётся (6, 4), а сумма каждой строки становится нулём.")
