import csv
import numpy as np
import matplotlib.pyplot as plt


# 1. Зчитування вхідних даних з CSV файлу
def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return np.array(x), np.array(y)


# 2. Формування матриці B (симетричної матриці нормальної системи МНК)
def form_matrix(x, m):
    size = m + 1
    B = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            B[i, j] = np.sum(x ** (i + j))
    return B


# 3. Формування вектора C (правої частини системи МНК)
def form_vector(x, y, m):
    size = m + 1
    C = np.zeros(size)
    for i in range(size):
        C[i] = np.sum(y * (x ** i))
    return C


# 4. Розв'язання СЛАР методом Гауса з вибором головного елемента по стовпцях
def gauss_solve(A_in, b_in):
    A = A_in.copy()
    b = b_in.copy()
    n = len(b)

    # Прямий хід з вибором головного елемента
    for k in range(n):
        # Вибір головного елемента в стовпці
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        # Виключення невідомих
        for i in range(k + 1, n):
            if A[k, k] == 0:
                continue
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотний хід
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = np.sum(A[i, i + 1:] * x_sol[i + 1:])
        if A[i, i] == 0:
            x_sol[i] = 0
        else:
            x_sol[i] = (b[i] - sum_ax) / A[i, i]

    return x_sol


# 5. Обчислення значення полінома для заданого масиву коефіцієнтів
def polynomial(x_vals, coef):
    y_poly = np.zeros_like(x_vals, dtype=float)
    for i, c in enumerate(coef):
        y_poly += c * (x_vals ** i)
    return y_poly


# 6. Обчислення дисперсії
def variance(y_true, y_approx):
    return np.sqrt(np.sum((y_approx - y_true) ** 2) / len(y_true))


# --- ГОЛОВНА ПРОГРАМА ---
if __name__ == "__main__":
    file_path = "data.csv"

    # Етап 1: Зчитування даних
    x, y = read_data(file_path)
    print("Вуси (Місяці x):", x)
    print("Значення (Температури y):", y)

    # Етап 2-3: Дослідження степенів від m = 1 до 10
    max_degree = 10
    variances = []
    all_coefficients = {}

    print("\n--- Дослідження дисперсії для різних степенів полінома ---")
    for m in range(1, max_degree + 1):
        B = form_matrix(x, m)
        C = form_vector(x, y, m)
        coef = gauss_solve(B, C)
        all_coefficients[m] = coef

        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        variances.append(var)
        print(f"Степінь m = {m:2d} | Дисперсія (δ) = {var:.4f}")

    # Вибір оптимального степеня (мінімальна дисперсія або розумний компроміс)
    optimal_m = int(np.argmin(variances) + 1)
    print(f"\n[Результат] Оптимальний степінь полінома за мінімумом дисперсії: m = {optimal_m}")

    # Етап 4: Прогноз на наступні 3 місяці (місяці 25, 26, 27)
    opt_coef = all_coefficients[optimal_m]
    x_future = np.array([25.0, 26.0, 27.0])
    y_future = polynomial(x_future, opt_coef)

    print("\n--- Прогноз температури на наступні 3 місяці ---")
    for month, temp in zip(x_future, y_future):
        print(f"Місяць {int(month)}: {temp:.2f} °C")

    # Етап 5: Побудова графіків
    # 1. Графік залежності дисперсії від степеня m
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_degree + 1), variances, marker='o', color='purple', linewidth=2)
    plt.axvline(x=optimal_m, color='red', linestyle='--', label=f'Оптимальний m = {optimal_m}')
    plt.title("Залежність дисперсії від степеня полінома")
    plt.xlabel("Степінь полінома (m)")
    plt.ylabel("Дисперсія (δ)")
    plt.grid(True)
    plt.legend()

    # 2. Графік апроксимації та фактичних даних разом із прогнозом
    plt.subplot(1, 2, 2)
    x_fine = np.linspace(x[0], x_future[-1], 200)
    y_fine = polynomial(x_fine, opt_coef)

    plt.scatter(x, y, color='blue', label='Фактичні дані (CSV)', zorder=5)
    plt.plot(x_fine, y_fine, color='red', label=f'Апроксимація (m={optimal_m})', linewidth=2)
    plt.scatter(x_future, y_future, color='green', s=100, label='Прогноз (3 місяці)', zorder=6)

    plt.title("МНК: Апроксимація та прогноз температур")
    plt.xlabel("Місяць")
    plt.ylabel("Температура (°C)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 3. Графік похибки апроксимації
    plt.figure(figsize=(8, 4))
    y_approx_opt = polynomial(x, opt_coef)
    errors = np.abs(y - y_approx_opt)
    plt.plot(x, errors, marker='s', color='orange', linewidth=2, label='Модуль похибки |f(x) - φ(x)|')
    plt.title("Графік похибки апроксимації у вузлах")
    plt.xlabel("Місяць")
    plt.ylabel("Похибка")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()