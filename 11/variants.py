"""
Таблиця варіантів практичної роботи №11 (Метаевристичні методи глобальної оптимізації).

Кожен рядок VARIANTS: (function_key, n, pop_size, alpha)
  function_key — ключ тестової функції у словнику FUNCTIONS нижче
  n            — розмірність задачі
  pop_size     — розмір популяції / рою (використовується і для ГА, і для PSO)
  alpha        — коефіцієнт геометричного охолодження Tk+1 = alpha*Tk для імітації відпалу

FUNCTIONS[key] описує саму тестову функцію:
  label   — назва (для підписів графіків/звіту)
  lo, hi  — межі області пошуку (однакові по всіх координатах)
  f(x)    — скалярна функція, x — 1-D numpy-масив довжини n
  f_grid(X, Y) — та сама функція, векторизована для n=2 (для контурних/поверхневих графіків)
  f_star(n)    — відоме значення глобального мінімуму (з довідкових джерел)
  x_star(n)    — відома точка глобального мінімуму (None, якщо немає простого замкненого вигляду)
"""
import numpy as np

# ---------- Тестові функції ----------


def _rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def _rastrigin_grid(X, Y):
    return 10 * 2 + (X ** 2 - 10 * np.cos(2 * np.pi * X)) + (Y ** 2 - 10 * np.cos(2 * np.pi * Y))


def _ackley(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2)))
        - np.exp(np.mean(np.cos(2 * np.pi * x)))
        + 20 + np.e
    )


def _ackley_grid(X, Y):
    return (
        -20 * np.exp(-0.2 * np.sqrt((X ** 2 + Y ** 2) / 2))
        - np.exp((np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)) / 2)
        + 20 + np.e
    )


def _griewank(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    i = np.arange(1, n + 1)
    return float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)


def _griewank_grid(X, Y):
    return (X ** 2 + Y ** 2) / 4000 - np.cos(X / np.sqrt(1)) * np.cos(Y / np.sqrt(2)) + 1


def _rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def _rosenbrock_grid(X, Y):
    return 100 * (Y - X ** 2) ** 2 + (1 - X) ** 2


def _schwefel(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def _schwefel_grid(X, Y):
    return 418.9829 * 2 - (X * np.sin(np.sqrt(np.abs(X))) + Y * np.sin(np.sqrt(np.abs(Y))))


def _michalewicz(x, m=10):
    x = np.asarray(x, dtype=float)
    n = len(x)
    i = np.arange(1, n + 1)
    return float(-np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m)))


def _michalewicz_grid(X, Y, m=10):
    return -(np.sin(X) * np.sin(1 * X ** 2 / np.pi) ** (2 * m) + np.sin(Y) * np.sin(2 * Y ** 2 / np.pi) ** (2 * m))


# відомі довідкові значення глобального мінімуму функції Міхалевича (m=10) для n = 2, 5, 10
_MICHALEWICZ_FSTAR = {2: -1.8013034101, 5: -4.6876581258, 10: -9.6601517447}

FUNCTIONS = {
    "rastrigin": dict(
        label="Растрігіна", lo=-5.12, hi=5.12,
        f=_rastrigin, f_grid=_rastrigin_grid,
        f_star=lambda n: 0.0, x_star=lambda n: np.zeros(n),
    ),
    "ackley": dict(
        label="Еклі", lo=-32.0, hi=32.0,
        f=_ackley, f_grid=_ackley_grid,
        f_star=lambda n: 0.0, x_star=lambda n: np.zeros(n),
    ),
    "griewank": dict(
        label="Грівонка", lo=-600.0, hi=600.0,
        f=_griewank, f_grid=_griewank_grid,
        f_star=lambda n: 0.0, x_star=lambda n: np.zeros(n),
    ),
    "rosenbrock": dict(
        label="Розенброка", lo=-5.0, hi=10.0,
        f=_rosenbrock, f_grid=_rosenbrock_grid,
        f_star=lambda n: 0.0, x_star=lambda n: np.ones(n),
    ),
    "schwefel": dict(
        label="Швефеля", lo=-500.0, hi=500.0,
        f=_schwefel, f_grid=_schwefel_grid,
        f_star=lambda n: 0.0, x_star=lambda n: np.full(n, 420.9687),
    ),
    "michalewicz": dict(
        label="Міхалевича", lo=0.0, hi=np.pi,
        f=_michalewicz, f_grid=_michalewicz_grid,
        f_star=lambda n: _MICHALEWICZ_FSTAR.get(n),
        x_star=lambda n: np.array([2.20319, 1.57049]) if n == 2 else None,
    ),
}

# (function_key, n, pop_size, alpha)
VARIANTS = {
    1:  ("rastrigin",    2,  20, 0.95),
    2:  ("ackley",       2,  20, 0.95),
    3:  ("griewank",     2,  20, 0.95),
    4:  ("rosenbrock",   2,  20, 0.95),
    5:  ("schwefel",     2,  20, 0.95),
    6:  ("michalewicz",  2,  20, 0.95),
    7:  ("rastrigin",    5,  25, 0.90),
    8:  ("ackley",       5,  25, 0.90),
    9:  ("griewank",     5,  25, 0.90),
    10: ("rosenbrock",   5,  25, 0.90),
    11: ("schwefel",     5,  25, 0.90),
    12: ("michalewicz",  5,  25, 0.90),
    13: ("rastrigin",    10, 30, 0.99),
    14: ("ackley",       10, 30, 0.99),
    15: ("griewank",     10, 30, 0.99),
    16: ("rosenbrock",   10, 30, 0.99),
    17: ("schwefel",     10, 30, 0.99),
    18: ("michalewicz",  10, 30, 0.99),
    19: ("rastrigin",    5,  35, 0.98),
    20: ("ackley",       5,  35, 0.98),
    21: ("griewank",     5,  35, 0.98),
    22: ("rosenbrock",   5,  35, 0.98),
    23: ("schwefel",     5,  35, 0.98),
    24: ("michalewicz",  5,  35, 0.98),
    25: ("rastrigin",    2,  40, 0.85),
    26: ("ackley",       2,  40, 0.85),
    27: ("griewank",     2,  40, 0.85),
    28: ("rosenbrock",   2,  40, 0.85),
    29: ("schwefel",     2,  40, 0.85),
    30: ("michalewicz",  2,  40, 0.85),
}
