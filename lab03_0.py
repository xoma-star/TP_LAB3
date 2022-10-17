# -*- coding: utf-8 -*-
import math
import time
import numpy
import matplotlib.pyplot


if __name__ == '__main__':
    # Точность
    eps = 1e-3
    # Шаги сетки по осям
    step_x = 1e-3
    step_y = 1e-3
    # Размерности прямоугольной области
    ix, jy = 0.1, 0.1

    print('Calculation started')
    td = time.time()
    
    # Невязка
    residual = []
    # Искомая функция
    res_f = lambda x, y: math.exp(-y * y / 4.0 - x * x / 16.0) * math.sin(x * x + y * y)

    # Элементы матрицы системы
    a1 = 1.0 / step_x / step_x
    an = 1.0 / step_y / step_y
    ai = -2.0 * (a1 + an)
    # Определение количества элементов по каждой из осей координат
    nx = int(ix / step_x) - 1
    ny = int(jy / step_y) - 1
    # Проверка на применимость метода
    if abs(ai) < eps:
        print('|a[i][i]| < eps')
        exit(1)

    # Вычисление размера матрицы
    n = nx * ny
    rnum = int((nx - 2) * (ny - 2) / 2 + (nx - 2) * (ny - 2) % 2)
    bnum = int((nx - 2) * (ny - 2) - rnum)

    # Значения искомой функции на текущей и предыдущей итерации
    u0 = [0] * nx * ny
    u1 = [0] * nx * ny
    # Правая часть уравнения Пуассона
    f = [0] * n
    # Номера узлов (для распределения переменных по шашечной схеме)
    red = [0] * rnum
    black = [0] * bnum

    # Заполнение
    for i in range(nx):
        x2 = (i * step_x) ** 2
        for j in range(ny):
            y2 = (j * step_y) ** 2
            x2y2 = x2 + y2
            sinx2y2 = math.sin(x2y2)
            cosx2y2 = math.cos(x2y2)

            f[i * ny + j] = (-1.0 / 64.0) \
							* math.exp(-(1.0 / 16.0) * x2 - (1.0 / 4.0) * y2) \
							* (\
							(255.0 * x2 + 240.0 * y2 + 40.0) * sinx2y2 \
							+ (32.0 * x2 + 128.0 * y2 + 256.0) * cosx2y2 \
							)

    # Заполнение
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u0[i * ny + j] = 0
            u1[i * ny + j] = 0
    # Установка граничных условий
    for j in range(nx):
        u0[j * ny] = res_f(j * step_x, 0)
        u0[(j + 1) * ny - 1] = res_f(j * step_x, jy - step_y)
        u1[j * ny] = res_f(j * step_x, 0)
        u1[(j + 1) * ny - 1] = res_f(j * step_x, jy - step_y)
    for i in range(1, ny - 1):
        u0[i] = res_f(0, i * step_y)
        u0[(nx - 1) * ny + i] = res_f(ix - step_x, i * step_y)
        u1[i] = res_f(0, i * step_y)
        u1[(nx - 1) * ny + i] = res_f(ix - step_x, i * step_y)

    # Выбор красных и чёрных узлов
    clr1 = False
    clr2 = False
    indr = 0
    indb = 0

    for i in range(1, nx - 1):
        clr1 = clr1 == 0
        clr2 = clr1

        for j in range(1, ny - 1):
            if clr2:
                red[indr] = i * ny + j
                indr += 1
            else:
                black[indb] = i * ny + j
                indb += 1
            clr2 = clr2 == 0

    # Решение системы методом Гаусса-Зейделя
    iter_num = 0
    diff = eps + 1
    while diff > eps:
        diff = 0
		# Вычисление значений в красных узлах
        for i in range(rnum):
            u1[red[i]] = -(a1 * (u0[red[i] - 1] + u0[red[i] + 1]) + an * (u0[red[i] + ny] + u0[red[i] - ny]) - f[red[i]]) / ai

		# Вычисление значений в чёрных узлах
        for i in range(bnum):
            u1[black[i]] = -(a1 * (u1[black[i] - 1] + u1[black[i] + 1]) + an * (u1[black[i] + ny] + u1[black[i] - ny]) - f[black[i]]) / ai

		# Вычисление разницы между текущей итерацией и следующей
        for i in range(n):
            diff += abs(u0[i] - u1[i])

		# Текущая итерация принимает значение следующей
        u0, u1 = u1, u0
        diff = math.sqrt(diff)
        residual.append(diff)
        print(diff)        

        iter_num += 1

    td = time.time() - td

    print(f'Time duration = {td}' )
    print(f'iter = {iter_num}')

    # Вывод полученного решения
    matplotlib.pyplot.figure()
    matplotlib.pyplot.subplot(1, 3, 1)
    matplotlib.pyplot.imshow(numpy.array(u0).reshape(nx, ny))
    matplotlib.pyplot.colorbar()

	# Вывод погрешности решения
    matplotlib.pyplot.subplot(1, 3, 2)
    matplotlib.pyplot.imshow(numpy.abs(numpy.array(u0) - [res_f(i * step_x, j * step_y) for i in range(nx) for j in range(ny)]).reshape(nx, ny))
    matplotlib.pyplot.colorbar()

    # Вывод графика невязки
    matplotlib.pyplot.subplot(1, 3, 3)
    matplotlib.pyplot.semilogy(residual, 'b-')
    matplotlib.pyplot.grid(True)

    matplotlib.pyplot.show()
