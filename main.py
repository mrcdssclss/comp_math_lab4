import numpy as np
import matplotlib.pyplot as plt
from math import log, exp, sqrt


#y = a + b*x
def linear_approximation(xs, ys, n):
    sx = sum(xs)
    sxx = sum(x ** 2 for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    A = np.array([[n, sx], [sx, sxx]])
    B = np.array([sy, sxy])
    a, b = np.linalg.solve(A, B)
    func = lambda x: a + b * x

    x_mean = sx / n
    y_mean = sy / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = sqrt(sum((x - x_mean) ** 2 for x in xs))
    den_y = sqrt(sum((y - y_mean) ** 2 for y in ys))
    r = num / (den_x * den_y) if den_x * den_y != 0 else 0

    y_pred = [func(x) for x in xs]
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return func, a, b, r, r2


#y = a + b*x + c*x^2
def quadratic_approximation(xs, ys, n):
    sx = sum(xs)
    sxx = sum(x ** 2 for x in xs)
    sxxx = sum(x ** 3 for x in xs)
    sxxxx = sum(x ** 4 for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxxy = sum(x * x * y for x, y in zip(xs, ys))
    A = np.array([[n, sx, sxx], [sx, sxx, sxxx], [sxx, sxxx, sxxxx]])
    B = np.array([sy, sxy, sxxy])
    a, b, c = np.linalg.solve(A, B)

    func = lambda x: a + b * x + c * x ** 2
    y_mean = sy / n
    y_pred = [func(x) for x in xs]
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return func, a, b, c, r2


#y = a + b*x + c*x^2 + d*x^3
def cubic_approximation(xs, ys, n):
    sx = sum(xs)
    sxx = sum(x ** 2 for x in xs)
    sxxx = sum(x ** 3 for x in xs)
    sxxxx = sum(x ** 4 for x in xs)
    sxxxxx = sum(x ** 5 for x in xs)
    sxxxxxx = sum(x ** 6 for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxxy = sum(x * x * y for x, y in zip(xs, ys))
    sxxxy = sum(x * x * x * y for x, y in zip(xs, ys))
    A = np.array([[n, sx, sxx, sxxx], [sx, sxx, sxxx, sxxxx],
                  [sxx, sxxx, sxxxx, sxxxxx], [sxxx, sxxxx, sxxxxx, sxxxxxx]])
    B = np.array([sy, sxy, sxxy, sxxxy])
    a, b, c, d = np.linalg.solve(A, B)

    func = lambda x: a + b * x + c * x ** 2 + d * x ** 3
    y_mean = sy / n
    y_pred = [func(x) for x in xs]
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return func, a, b, c, d, r2


# y = a * exp(b*x)
def exponential_approximation(xs, ys, n):
    ys_log = [log(y) for y in ys]
    func, a_log, b, _, _ = linear_approximation(xs, ys_log, n)
    a = exp(a_log)

    func = lambda x: a * exp(b * x)
    sy = sum(ys)
    y_mean = sy / n
    y_pred = [func(x) for x in xs]
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return func, a, b, r2

# y = a + b * ln(x)
def logarithmic_approximation(xs, ys, n):
    xs_log = [log(x) for x in xs]
    func, a, b, _, _ = linear_approximation(xs_log, ys, n)

    func = lambda x: a + b * log(x)
    sy = sum(ys)
    y_mean = sy / n
    y_pred = [func(x) for x in xs]
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return func, a, b, r2


#y = a * x^b
def power_approximation(xs, ys, n):
    xs_log = [log(x) for x in xs]
    ys_log = [log(y) for y in ys]
    func, a_log, b, _, _ = linear_approximation(xs_log, ys_log, n)
    a = exp(a_log)

    func = lambda x: a * x ** b
    sy = sum(ys)
    y_mean = sy / n
    y_pred = [func(x) for x in xs]
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - y_pred[i]) ** 2 for i, y in enumerate(ys))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return func, a, b, r2

def compute_mse(xs, ys, func, n):
    return sqrt(sum((func(x) - y) ** 2 for x, y in zip(xs, ys)) / n),  sum((func(x) - y) ** 2 for x, y in zip(xs, ys))


def compute_values(func, xs):
    return [func(x) for x in xs]

def draw_best_func(func, name, xs, ys):
    x_min, x_max = min(xs) - 0.1 * (max(xs) - min(xs)), max(xs) + 0.1 * (max(xs) - min(xs))
    y_min, y_max = min(ys) - 0.1 * (max(ys) - min(ys)), max(ys) + 0.1 * (max(ys) - min(ys))
    x = np.linspace(x_min, x_max, 100)
    y = [func(xi) for xi in x]
    plt.scatter(xs, ys, label="Данные", color="blue")
    plt.plot(x, y, label=f"Аппроксимация: {name}", color="red")
    plt.title("Лучшая аппроксимация")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)
    plt.show()


def read_from_file():
    filename = input("Введите имя файла: ")
    try:
        with open(filename, 'r') as f:
            xs, ys = [], []
            for line in f:
                x, y = map(float, line.strip().split())
                xs.append(x)
                ys.append(y)
        if not (8 <= len(xs) <= 12):
            raise ValueError("Количество точек должно быть от 8 до 12")
        if len(xs) != len(set(xs)):
            raise ValueError("Значения x должны быть уникальными")
        return xs, ys
    except Exception as e:
        print(f"Ошибка чтения файла: {e}")
        return None, None


def read_from_keyboard():
    xs, ys = [], []
    print("Введите от 8 до 12 точек (x y), для завершения введите 'стоп':")
    while len(xs) < 12:
        inp = input(f"Точка {len(xs) + 1}: ").strip()
        if inp.lower() == "стоп" and len(xs) >= 8:
            break
        elif inp.lower() == "стоп":
            print("Введите минимум 8 точек")
            continue
        try:
            x, y = map(float, inp.split())
            if x in xs:
                print("Значения x должны быть уникальными")
                continue
            xs.append(x)
            ys.append(y)
        except:
            print("Неверный формат, введите два числа через пробел")
    return xs, ys

def get_func_str(coeffs, name):
    if name == "Линейная":
        a, b = coeffs
        return f"{a:.4f} + {b:.4f}*x"
    elif name == "Квадратичная":
        a, b, c = coeffs
        return f"{a:.4f} + {b:.4f}*x + {c:.4f}*x^2"
    elif name == "Кубическая":
        a, b, c, d = coeffs
        return f"{a:.4f} + {b:.4f}*x + {c:.4f}*x^2 + {d:.4f}*x^3"
    elif name == "Экспоненциальная":
        a, b = coeffs
        return f"{a:.4f} * exp({b:.4f}*x)"
    elif name == "Логарифмическая":
        a, b = coeffs
        return f"{a:.4f} + {b:.4f}*ln(x)"
    elif name == "Степенная":
        a, b = coeffs
        return f"{a:.4f} * x^{b:.4f}"

def run(xs, ys, n):
    functions = [
        (linear_approximation, "Линейная"),
        (quadratic_approximation, "Квадратичная"),
        (cubic_approximation, "Кубическая")
    ]
    if all(x > 0 for x in xs):
        if all(y > 0 for y in ys):
            functions.extend([
                (exponential_approximation, "Экспоненциальная"),
                (power_approximation, "Степенная")
            ])
        functions.append((logarithmic_approximation, "Логарифмическая"))

    best_mse = float("inf")
    best_func = None
    best_name = ""
    results = []

    print("\nРезультаты аппроксимации:")
    for approx, name in functions:
        try:
            result = approx(xs, ys, n)
            func = result[0]
            coeffs = result[1:3] if name in ["Линейная", "Экспоненциальная", "Логарифмическая", "Степенная"] else result[1:-1]
            r2 = result[-1]  # R^2 всегда последний элемент
            mse, sred = compute_mse(xs, ys, func, n)
            values = compute_values(func, xs)
            eq_str = get_func_str(coeffs, name)
            results.append((name, eq_str, mse, values, r2))
            print(f"{name}:")
            print(f"  Уравнение: f(x) = {eq_str}")
            print(f"  Среднеквадратичное отклонение: {mse:.5f}")
            if 1 >= r2 > 0.95:
                print(f"  Коэффициент детерминации: {r2:.5f} - высокая точность аппроксимации!")
            elif 0.95 >= r2 > 0.75:
                print(f"  Коэффициент детерминации: {r2:.5f} - удовлетворительная точность аппроксимации!")
            elif 0.75 >= r2 > 0.5:
                print(f"  Коэффициент детерминации: {r2:.5f} - слабая точность аппроксимации!")
            else:
                print(f"  Коэффициент детерминации: {r2:.5f} - точность аппроксимации недостаточна!")
            print(f"  Значения функции: {', '.join(f'{v:.4f}' for v in values)}")
            if name == "Линейная":
                r = result[3]
                print(f"  Коэффициент корреляции Пирсона: {r:.5f}")
            print()
            if mse < best_mse:
                best_mse = mse
                best_func = func
                best_name = name
        except Exception as e:
            print(f"Ошибка в {name}: {e}\n")

    if best_func:
        print(f"Лучшая аппроксимация: {best_name}")
        print(f"Среднеквадратичная ошибка: {best_mse:.5f}, {sred:.5f}")
        draw_best_func(best_func, best_name, xs, ys)
    else:
        print("Не удалось выполнить аппроксимацию для данных точек")

def main():
    while True:
        try:
            choice = int(input("Выберите источник данных: 1. Файл 2. Клавиатура: "))
            if choice not in [1, 2]:
                print("Неверный выбор, выберите 1 или 2")
                continue
            break
        except ValueError:
            print("Введите число (1 или 2)")

    if choice == 1:
        xs, ys = read_from_file()
        if xs is None:
            return
    else:
        xs, ys = read_from_keyboard()

    n = len(xs)
    run(xs, ys, n)

main()