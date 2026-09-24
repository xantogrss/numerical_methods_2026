import csv
import numpy as np
import matplotlib.pyplot as plt


# 1. Функція для зчитування даних з CSV файлу
def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return np.array(x), np.array(y)


# 2. Побудова таблиці розділених різниць (повертає коефіцієнти полінома Ньютона)
def divided_differences_table(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])

    return coef[0, :]


# 3. Обчислення значення інтерполяційного многочлена Ньютона у точці x_val
def newton_polynomial(x_nodes, coef, x_val):
    n = len(x_nodes)
    p = coef[0]
    product = 1.0
    for i in range(1, n):
        product *= (x_val - x_nodes[i - 1])
        p += coef[i] * product
    return p


# --- ГОЛОВНА ПРОГРАМА ---
if __name__ == "__main__":
    # Шлях до файлу з даними
    file_path = "data.csv"

    # Крок 1: Зчитуємо дані
    x_nodes, y_nodes = read_data(file_path)
    print("Вузли n (розмір даних):", x_nodes)
    print("Значення t (час виконання):", y_nodes)

    # Крок 2: Будуємо таблицю розділених різниць
    ny_coef = divided_differences_table(x_nodes, y_nodes)
    print("\nКоефіцієнти полінома Ньютона:")
    for i, c in enumerate(ny_coef):
        print(f"f(x0...x{i}) = {c}")

    # Крок 3: Прогнозування часу виконання при n = 6000
    x_target = 6000
    t_pred = newton_polynomial(x_nodes, ny_coef, x_target)
    print(f"\n[Прогноз] Час виконання для n = {x_target}: {t_pred:.4f} мс")

    # Крок 4: Табуляція для побудови плавної кривої на графіку
    x_fine = np.linspace(x_nodes[0], x_nodes[-1], 300)
    y_fine = [newton_polynomial(x_nodes, ny_coef, xv) for xv in x_fine]

    # Крок 5: Побудова графіків
    plt.figure(figsize=(9, 6))
    plt.plot(x_fine, y_fine, label="Інтерполяційна крива Ньютона", color="red", linewidth=2)
    plt.scatter(x_nodes, y_nodes, color="blue", s=60, zorder=5, label="Експериментальні точки")
    plt.scatter([x_target], [t_pred], color="green", s=120, zorder=6, label=f"Прогноз n={x_target} ({t_pred:.2f} мс)")

    plt.title("Лабораторна робота №2: Інтерполяція многочленом Ньютона (Варіант 1)")
    plt.xlabel("Розмір вхідних даних (n)")
    plt.ylabel("Час виконання (мс)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()