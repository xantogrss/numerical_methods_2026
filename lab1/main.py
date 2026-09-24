import numpy as np
import matplotlib.pyplot as plt


# Визначаємо трансцендентну функцію f(x), наприклад, sin(x)
def f(x):
    return np.sin(x)


# ЕТАП 1: Табуляція вхідної функції
x_initial = 0.0
x_final = np.pi
n = 25  # Кількість вузлів (в межах 20..30)
h = (x_final - x_initial) / n

List_X = [x_initial + i * h for i in range(n + 1)]
List_Y = [f(x) for x in List_X]

# Збереження вузлів у файл "вузли.txt"
with open("вузли.txt", "w", encoding="utf-8") as f_out:
    f_out.write("X\tY\n")
    for xi, yi in zip(List_X, List_Y):
        f_out.write(f"{xi:.6f}\t{yi:.6f}\n")

print("Етап 1 пройдено: вузли збережено у файл 'вузли.txt'.")

# ЕТАП 2: Підготовка системи рівнянь для коефіцієнтів c_i
List_H = [List_X[i] - List_X[i - 1] for i in range(1, n + 1)]

# Вектори для трьохдіагональної матриці
List_Альфа = [0.0] * (n + 1)
List_Бета = [0.0] * (n + 1)
List_Гамма = [0.0] * (n + 1)
List_Дельта = [0.0] * (n + 1)

# Умови для вільних країв
List_Альфа[1] = 0.0
List_Бета[1] = 1.0
List_Гамма[1] = 0.0
List_Дельта[1] = 0.0

for i in range(2, n):
    h_prev = List_H[i - 1]
    h_curr = List_H[i]

    ПраваЧастина = 3.0 * ((List_Y[i] - List_Y[i - 1]) / h_curr - (List_Y[i - 1] - List_Y[i - 2]) / h_prev)

    List_Альфа[i] = h_prev
    List_Бета[i] = 2.0 * (h_prev + h_curr)
    List_Гамма[i] = h_curr
    List_Дельта[i] = ПраваЧастина

# Крайова умова для n-го вузла (виправлена індексація)
h_n_prev = List_H[n - 2]
h_n = List_H[n - 1]
ПраваЧастина_n = 3.0 * ((List_Y[n] - List_Y[n - 1]) / h_n - (List_Y[n - 1] - List_Y[n - 2]) / h_n_prev)

List_Альфа[n] = h_n_prev
List_Бета[n] = 2.0 * (h_n_prev + h_n)
List_Гамма[n] = 0.0
List_Дельта[n] = ПраваЧастина_n

print("Етап 2 пройдено: коефіцієнти матриці сформовано.")


# ЕТАП 3: Метод прогонки
def MetodProgonki(n_sys, alfa, beta, gamma, delta):
    A = [0.0] * (n_sys + 1)
    B = [0.0] * (n_sys + 1)
    X_rozv = [0.0] * (n_sys + 1)

    A[1] = -gamma[1] / beta[1]
    B[1] = delta[1] / beta[1]

    for i in range(2, n_sys):
        denominator = alfa[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denominator
        B[i] = (delta[i] - alfa[i] * B[i - 1]) / denominator

    X_rozv[n_sys] = (delta[n_sys] - alfa[n_sys] * B[n_sys - 1]) / (alfa[n_sys] * A[n_sys - 1] + beta[n_sys])

    for i in range(n_sys - 1, 0, -1):
        X_rozv[i] = A[i] * X_rozv[i + 1] + B[i]

    return X_rozv


List_C = MetodProgonki(n, List_Альфа, List_Бета, List_Гамма, List_Дельта)
print("Етап 3 пройдено: метод прогонки виконано.")

# ЕТАП 4: Відновлення решти коефіцієнтів (a, b, d)
List_A = [0.0] * (n + 1)
List_B = [0.0] * (n + 1)
List_D = [0.0] * (n + 1)

for i in range(1, n + 1):
    List_A[i] = List_Y[i - 1]

    h_curr = List_H[i - 1]
    if i < n:
        List_D[i] = (List_C[i + 1] - List_C[i]) / (3.0 * h_curr)
        List_B[i] = (List_Y[i] - List_Y[i - 1]) / h_curr - (h_curr / 3.0) * (List_C[i + 1] + 2.0 * List_C[i])
    else:
        List_D[i] = -List_C[n] / (3.0 * h_curr)
        List_B[i] = (List_Y[n] - List_Y[n - 1]) / h_curr - (2.0 / 3.0) * h_curr * List_C[n]

print("Етап 4 пройдено: коефіцієнти a, b, d розраховано.")

# ЕТАП 5: Фінальна табуляція та підготовка до графіків
N_tab = 20 * n
x_fine = np.linspace(x_initial, x_final, N_tab)
y_exact = f(x_fine)
s_approx = []

for x_val in x_fine:
    j = 1
    for idx in range(1, n + 1):
        if List_X[idx - 1] <= x_val <= List_X[idx]:
            j = idx
            break
    dx = x_val - List_X[j - 1]
    s_val = List_A[j] + List_B[j] * dx + List_C[j] * (dx ** 2) + List_D[j] * (dx ** 3)
    s_approx.append(s_val)

s_approx = np.array(s_approx)
errors = np.abs(y_exact - s_approx)

print("Етап 5 пройдено: похибки обчислено.")

# Побудова графіків
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x_fine, y_exact, label="Точна функція f(x)", color="blue", linewidth=2)
plt.plot(x_fine, s_approx, label="Кубічний сплайн S(x)", color="red", linestyle="--", linewidth=2)
plt.scatter(List_X, List_Y, color="black", zorder=5, label="Вузли табуляції")
plt.legend()
plt.title("Інтерполяція кубічними сплайнами")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_fine, errors, label="Похибка |y - S(x)|", color="green")
plt.legend()
plt.title("Графік похибки")
plt.xlabel("x")
plt.grid(True)

plt.tight_layout()
plt.show()