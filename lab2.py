import sympy as sp
from scipy import signal
import control as co
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import TransferFunction, StateSpace


s = sp.Symbol('s', rational=True)

def multiply_blocks(num_list, den_list):
    numerator = 1
    denominator = 1

    for num in num_list:
        numerator *= sp.Poly(num, s)

    for den in den_list:
        denominator *= sp.Poly(den, s)

    return numerator, denominator

def check_stability(num, den):
    H = sp.Poly(num, s) / sp.Poly(den, s)
    poles = sp.roots(den, s)
    zeros = sp.roots(num, s)

    is_stable = all(sp.re(p) < 0 for p in poles)

    return poles, zeros, is_stable

num_blocks = int(input("Введите количество блоков в системе управления: "))
num_list = []
den_list = []

for i in range(num_blocks):
    print(f"Блок № {i + 1}")
    num = input("Введите значения(коэффициенты) числителя через пробел: ").split()
    den = input("Введите значения(коэффициенты) знаменателя через пробел: ").split()
    num_list.append([float(n) for n in num])
    den_list.append([float(d) for d in den])

s = sp.symbols('s')
numerator, denominator = multiply_blocks(num_list, den_list)

forward_path = sp.Poly(numerator, s) / sp.Poly(denominator, s)

print("Передаточная функция прямого пути :", forward_path)


num_feedback_blocks = int(input("Введите количество блоков обратой связи: "))
num_feedback_list = []
den_feedback_list = []

for i in range(num_feedback_blocks):
    print(f"Блок обратной связи № {i + 1}")
    num_feedback = input("Введите значения(коэффициенты) числителя обратной связи через пробел: ").split()
    den_feedback = input("EВведите значения(коэффициенты) знаменателя обратной связи через пробел: ").split()
    num_feedback_list.append([float(n) for n in num_feedback])
    den_feedback_list.append([float(d) for d in den_feedback])
    
s = sp.symbols('s')
numerator_feedback, denominator_feedback = multiply_blocks(num_feedback_list, den_feedback_list)
feedback_path = sp.Poly(numerator_feedback, s) / sp.Poly(denominator_feedback, s)

feedback_type = input("Определите тип обратной связи:\n1. Отрицательная\n2. Положительная\n")


# Расчет передаточной функции с обратной свзяью
if feedback_type == '1':
    transfer_function = forward_path / (1 + (forward_path * feedback_path))
elif feedback_type == '2':
    transfer_function = forward_path / (1 - (forward_path * feedback_path))
else:
    print("Неправильно указан тип обратной связи!")

G=transfer_function.simplify()
print("Результирующая передаточная функция:")
sp.pprint(G)
numerator_final, denominator_final = sp.fraction(G)
top, bot = [[float(i) for i in sp.Poly(i, s).all_coeffs()] for i in G.as_numer_denom()]
sys=co.TransferFunction(top, bot)
plt.figure(1)
plt.figure(figsize=(6, 4), dpi=100)
co.pzmap(sys, grid=True)
plt.title('Комплексная плоскость Нулей и Полюсов системы')
plt.show()

poles, zeros, is_stable = check_stability(numerator_final, denominator_final)
print("Полюса:", poles)
print("Нули:", zeros)
print("Соответственно положению полюсов на комплексной плоскости система: ", "Стабильна" if is_stable else "Нестабильна")

def nyquist_criterion(num, den):
    # Определение границ частоты в бесконечность + и -
    w = np.logspace(-3, 3, num=1000)

    # Расчет передаточной функции для каждой частоты
    s = 1j * w
    G = np.polyval(num, s) / np.polyval(den, s)

    # Расчет магнитуды и фазы от G
    mag = np.abs(G)
    phase = np.angle(G)

    # Годограф Найквиста
    plt.figure(2)
    plt.plot(np.real(G), np.imag(G))
    plt.plot(np.real(G[0]), np.imag(G[0]), 'ro')  # Точка отчета
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Годограф Найквиста')
    plt.grid(True)

    # Построение запаса устойчивости
    R = np.max(mag)
    theta = np.linspace(0, 2 * np.pi, num=100)
    re_limit = np.cos(theta)
    im_limit = np.sin(theta)
    plt.plot(re_limit, im_limit, 'r--', label='Предел устойчивости')
    plt.legend()

    plt.show()

    # Определение устойчивости по Найквистуу
    crosses_origin = np.sum(np.diff(np.sign(mag)))  # Сколько раз кривая пересекла черту

    if crosses_origin == 0:
        print("Соответственно критерию Найквиста, система устойчива.")
    elif crosses_origin < 0:
        print("Соответственно критерию Найквиста, система неустойчива.")
    else:
        print("Соответственно критерию Найквиста, система на границе устойчивости.")

nyquist_criterion(top, bot)


def create_hurwitz_matrix(coefficients):
    k = 0
    matrix = []
    for _ in range(0, len(coefficients)-1):
        column = []
        for d in range(0, len(coefficients)-1):
            if 2*d+1-k < 0:
                column.append(0)
            else:
                try:
                    column.append(coefficients[2*d+1-k])
                except IndexError:
                    column.append(0)
            d += 1
        matrix.append(column)
        k += 1
    return np.array(matrix)

#def check_matrix_determinant(matrix):
#   check if the matrix denominator renders the system unstable or marginally
#   stable

def create_minor(matrix, row, column):
    # Убираем i-столбец и j-столбец
    return matrix[
        np.array(list(range(row)) +
                 list(range(row+1, matrix.shape[0])))[:,np.newaxis],
        np.array(list(range(column)) +
                 list(range(column+1, matrix.shape[1])))]

def check_stability(hurwitz_matrix, coefficients):
    a=0
    print(hurwitz_matrix)
    print("Определитель матрицы: "+
          str(np.linalg.det(hurwitz_matrix)))
    if np.linalg.det(hurwitz_matrix) > 0:
        print("Определитель матрицы больше нуля")
    elif np.linalg.det(hurwitz_matrix) == 0:
        print("Определитель матрицы равен нулю, система на границе устойчивости")
    else:
        return("Определитель матрицы меньше нуля, система неустойчива")

    for _ in range(0, len(coefficients)-2):
        x,y = hurwitz_matrix.shape
        hurwitz_matrix= create_minor(np.array(hurwitz_matrix), x-1, y-1)
        hurwitz_matrix= create_minor(hurwitz_matrix, x-1, y-1)
        print(hurwitz_matrix)
        print("Определитель матрицы: " +
              str(np.linalg.det(hurwitz_matrix)))
        if np.linalg.det(hurwitz_matrix) > 0:
            continue
        elif np.linalg.det(hurwitz_matrix) == 0:
            print("Определитель матрицы равен нулю, система на границе устойчивости")
            continue
        else:
            print("Определитель матрицы меньше нуля, система неустойчива")

    print("Определители матрицы больше нуля, система устойчива")


def create_minor(matrix, row, column):

    return matrix[np.array(list(range(row))+list(range(row+1, matrix.shape[0])))[:,np.newaxis],
               np.array(list(range(column))+list(range(column+1, matrix.shape[1])))]

coefficients = bot


print("Для устойчивости все определители по Гурвицу должны быть больше нуля")
matrix=create_hurwitz_matrix(coefficients)

a=0
newmatrix=np.array(matrix)
for _ in range(0, len(coefficients)-2):
    x,y = newmatrix.shape
    newmatrix= create_minor(np.array(newmatrix), x-1, y-1)
    print(newmatrix)
    print("Определитель данной матрицы равен:" +
          str(np.linalg.det(newmatrix)))
    if np.linalg.det(newmatrix) > 0:
        continue
    elif np.linalg.det(newmatrix) == 0:
        print("Система на границе устойчивости")
        continue
    else:
        print("Система неустойчива")
print("Система устойчива.")


print('Матрица Гурвица: \n')
newmatrix=matrix
print(check_stability(newmatrix, coefficients))



# Определение передаточной функции

tf = TransferFunction(top, bot)

# Перевод передаточой функции к графическому виду to state-space
ss = StateSpace(tf)

# Вычисление параметров функции
eig = np.linalg.eigvals(ss.A)

# Проверка устойчивости
if np.all(np.real(eig) < 0):
    print("Система устойчива")
else:
    print("Система неустойчива")

# Построение частотного графика характеристик
w, mag, phase = tf.bode()
plt.figure(4)
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.semilogx(w, mag)
ax1.set_ylabel("Магнитуда")
ax2.semilogx(w, phase)
ax2.set_xlabel("Частота (рад/с)")
ax2.set_ylabel("Фаза (deg)")
plt.show()


coeffVector = np.array(bot)
coeffLength = len(coeffVector)
rhTableColumn = int(np.ceil(coeffLength / 2))

# Запуск таблицы Раусса-Гурвица с нулями
rhTable = np.zeros((coeffLength, rhTableColumn))

# Расчет первой строки таблицы
rhTable[0, :] = coeffVector[::2]

# Проверка на четность длины вектора коэффициента
if coeffLength % 2 != 0:
    # Нечетная, вторая строка
    rhTable[1, :rhTableColumn - 1] = coeffVector[1::2]
else:
    # Четная, вторая строка
    rhTable[1, :] = coeffVector[1::2]

# Расчет строк
epss = 0.01

for i in range(2, coeffLength):
    # Особый случай: строка с нулями
    if np.all(rhTable[i - 1, :] == 0):
        order = coeffLength - i
        cnt1 = 0
        cnt2 = 1
        for j in range(rhTableColumn - 1):
            rhTable[i - 1, j] = (order - cnt1) * rhTable[i - 2, cnt2]
            cnt1 += 2
            cnt2 += 1

    for j in range(rhTableColumn - 1):
        # Расчет значений в таблице
        firstElemUpperRow = rhTable[i - 1, 0]
        rhTable[i, j] = ((rhTable[i - 1, 0] * rhTable[i - 2, j + 1]) - (
                    rhTable[i - 2, 0] * rhTable[i - 1, j + 1])) / firstElemUpperRow

        # Особый случай: ноль в первом столбце
        if rhTable[i, 0] == 0:
            rhTable[i, 0] = epss

# Подсчет полюсов справа от оси
unstablePoles = 0

for i in range(coeffLength - 1):
    if np.sign(rhTable[i, 0]) * np.sign(rhTable[i + 1, 0]) == -1:
        unstablePoles += 1

# Вывод полученных значений
print("\nТаблица Рауса-Гурвица:")
print(rhTable)

# Вывод результата об устойчивости
if unstablePoles == 0:
    print("~~~~~> Система устойчива! <~~~~~")
else:
    print("~~~~~> Система неустойчива! <~~~~~")

print("\nКоличество полюсов справа: %d" % unstablePoles)

sysRoots = np.roots(coeffVector)
print("\nПолученные корни уравнения:")
# Полученные корни уравнения
print(sysRoots)

def mikhailova_stability(numerator_coeffs, denominator_coeffs):
    # Получение полюсов передаточной функции
    poles = np.roots(denominator_coeffs)

    # Подсчет кол-ва полюсов с правой стороны плоскости
    rhp_poles_count = sum(np.real(poles) > 0)

    # Проверка устойчивости
    if rhp_poles_count == 0:
        return "Устойчива"
    elif rhp_poles_count == 1:
        return "На границе устойчивости"
    else:
        return "Неустойчива"

stability_result = mikhailova_stability(top, bot)
print("По критерию Михайлова система: ", stability_result)