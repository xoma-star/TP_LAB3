# -*- coding: utf-8 -*-
import math
import sys
import time
import numpy
import matplotlib.pyplot
import numpy as np
from numba import jit
#from pybind import gzWrapper

def GaussZeidel(u1, u0, f):
    residual = []
    iter_num = 0
    diff = eps + 1
    u1_ = u1[:]
    while diff > eps:
        diff = 0
        # Вычисление значений в узлах
        u1[1:-1, 1:-1] = -(an * (u0[:-2, 1:-1] + u0[2:, 1:-1]) + a1 * (u0[1:-1, :-2] + u0[1:-1, 2:]) - f[1:-1, 1:-1]) / ai
        # Текущая итерация принимает значение следующей
        u0, u1 = u1, u0
        # Вычисление разницы между текущей итерацией и следующей
        diff = np.sqrt(np.abs(u0 - u1).sum())
        residual.append(diff)
        #print(f'Difference: {diff}')
        print(diff)

        iter_num += 1
    return iter_num, u0, residual


def res_f(x, y):
    return np.exp(-np.power(y, 2) / 4.0 - np.power(x, 2) / 16.0) * np.sin(np.power(x, 2) + np.power(y, 2))


def FillMatrix(x, y, y2, x2, u0, u1, f):
    # Заполнение
    f = (-1.0 / 64.0) * np.exp(-(1.0 / 16.0) * x2 - (1.0 / 4.0) * y2) \
    * (
            (255.0 * x2 + 240.0 * y2 + 40.0) * np.sin(x2 + y2)
            + (32.0 * x2 + 128.0 * y2 + 256.0) * np.cos(x2 + y2)
    )
    # Заполнение
    u0[:] = 0
    u1[:] = 0

    # Установка граничных условий

    u0[:, 0] = u1[:, 0] = res_f(x, y[0])
    u0[:, -1] = u1[:, -1] = res_f(x, y[-1])
    u0[0, :] = u1[0, :] = res_f(x[0], y)
    u0[-1, :] = u1[-1, :] = res_f(x[-1], y)
    return  u0, u1, f

if __name__ == '__main__':
    # Точность
    eps = 1e-3
    # Шаги сетки по осям
    step_x = 0.001
    step_y = 0.001
    # Размерности прямоугольной области
    ix, jy = 0.2, 0.3


    print('Calculation started')
    td = time.time()

    # Невязка
    residual = []

    # Элементы матрицы системы
    a1 = 1.0 / step_x / step_x
    an = 1.0 / step_y / step_y
    ai = -2.0 * (a1 + an)
    # Определение количества элементов по каждой из осей координат
    nx = int(ix / step_x) - 1
    ny = int(jy / step_y) - 1
    print(f'nx: {nx}; ny: {ny}')
    # Проверка на применимость метода
    if abs(ai) < eps:
        print('|a[i][i]| < eps\n')
        print('!!!Method will diverge!!!')
        exit(1)

    # Вычисление размера матрицы
    n = nx * ny

    # Значения искомой функции на текущей и предыдущей итерации
    u0 = np.zeros((nx, ny), dtype=np.float32)
    u1 = np.zeros((nx, ny), dtype=np.float32)
    # Правая часть уравнения Пуассона
    f = np.zeros(n, dtype=float)

    x = np.linspace(0, ix, nx)
    y = np.linspace(0, jy, ny)
    y2, x2 = np.meshgrid(np.power(y, 2), np.power(x, 2))

    u0, u1, f = FillMatrix(x, y, y2, x2, u0, u1, f)

    # Решение системы методом Гаусса-Зейделя
    iter_num, u0, residual = GaussZeidel(u1, u0, f)
    #iter_num, u0, u1, residual = gzWrapper.GaussZeidel(u0, u1, red, black, f, n, ny, bnum, rnum, eps, a1, an, ai)

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
    yy, xx = np.meshgrid(y, x)
    matplotlib.pyplot.imshow(np.abs(u0 - res_f(xx, yy)))
    matplotlib.pyplot.colorbar()

    # Вывод графика невязки
    matplotlib.pyplot.subplot(1, 3, 3)
    matplotlib.pyplot.semilogy(residual, 'b-')
    matplotlib.pyplot.grid(True)

    matplotlib.pyplot.show()
