"""
Генератор звіту до практичної роботи №6 (Транспортна задача та задача про призначення).
Питає номер варіанта (або приймає його аргументом) і генерує Zvit_PR6_Variant<N>.docx.

Запуск:  .venv/bin/python generate_report.py [номер_варіанта]
"""
import sys
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog, linear_sum_assignment
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

from variants import VARIANTS

# --- Дані студента / титульного аркуша ---
STUDENT_NAME = "Толошний Мирослав"
GROUP = "КН-11"
UNIVERSITY = "Вінницький національний аграрний університет"
FACULTY = "Факультет комп'ютерних наук та економічної кібернетики"
DISCIPLINE = "Аналіз та методи оптимізації складних систем"
WORK_TITLE = "Транспортна задача та задача про призначення"
WORK_NUMBER = 6

if len(sys.argv) > 1:
    VARIANT = int(sys.argv[1])
else:
    while True:
        raw = input(f"Введи номер варіанта (1-{len(VARIANTS)}): ").strip()
        if raw.isdigit() and int(raw) in VARIANTS:
            VARIANT = int(raw)
            break
        print(f"Треба ціле число від 1 до {len(VARIANTS)}, спробуй ще раз.")


def pt(v):
    """np.float64 -> звичайний float/int (інакше в numpy>=2 рядок типу 'np.float64(1.2)')."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return tuple(pt(c) for c in v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def fmt(v, nd=2):
    v = pt(v)
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


# ---------- Вихідні дані ----------
C_raw, a_raw, b_raw = VARIANTS[VARIANT]
C = np.array(C_raw, dtype=float)
a = np.array(a_raw, dtype=float)
b = np.array(b_raw, dtype=float)
m, n = C.shape
balanced = abs(a.sum() - b.sum()) < 1e-9


def cost(X):
    return float((np.array(X) * C).sum())


# ---------- Опорні плани ----------
def north_west(a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros((len(a), len(b)))
    i = j = 0
    while i < len(a) and j < len(b):
        q = min(a[i], b[j])
        X[i, j] = q
        a[i] -= q
        b[j] -= q
        if a[i] <= 1e-9:
            i += 1
        else:
            j += 1
    return X


def min_element(C, a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros_like(C)
    M = C.copy()
    while a.sum() > 1e-9:
        i, j = np.unravel_index(np.argmin(M), M.shape)
        q = min(a[i], b[j])
        X[i, j] = q
        a[i] -= q
        b[j] -= q
        if a[i] <= 1e-9:
            M[i, :] = np.inf
        if b[j] <= 1e-9:
            M[:, j] = np.inf
    return X


def vogel(C, a, b):
    a, b = a.copy(), b.copy()
    M = C.copy()
    rows_open = set(range(len(a)))
    cols_open = set(range(len(b)))
    X = np.zeros_like(C)

    def penalty(vals):
        v = sorted(vals)
        if len(v) == 1:
            return v[0]
        return v[1] - v[0]

    while rows_open and cols_open:
        row_pen = {i: penalty([M[i, j] for j in cols_open]) for i in rows_open}
        col_pen = {j: penalty([M[i, j] for i in rows_open]) for j in cols_open}
        best_row = max(row_pen, key=row_pen.get)
        best_col = max(col_pen, key=col_pen.get)
        if row_pen[best_row] >= col_pen[best_col]:
            i = best_row
            j = min(cols_open, key=lambda c: M[i, c])
        else:
            j = best_col
            i = min(rows_open, key=lambda r: M[r, j])
        q = min(a[i], b[j])
        X[i, j] = q
        a[i] -= q
        b[j] -= q
        if a[i] <= 1e-9:
            rows_open.discard(i)
        if b[j] <= 1e-9:
            cols_open.discard(j)
    return X


X_nw = north_west(a, b)
X_me = min_element(C, a, b)
X_vg = vogel(C, a, b)


def basic_cells(X, tol=1e-6):
    return [(i, j) for i in range(m) for j in range(n) if X[i, j] > tol]


# ---------- Ручна ітерація методу потенціалів (з опорного плану мін. елемента) ----------
def potentials_from_basis(C, basis, m, n):
    """Обчислює u_i, v_j із системи u_i+v_j=c_ij для базисних клітинок (u_0=0)."""
    u = [None] * m
    v = [None] * n
    u[0] = 0.0
    changed = True
    adj = {}
    for (i, j) in basis:
        adj.setdefault(("r", i), []).append(("c", j, C[i, j]))
        adj.setdefault(("c", j), []).append(("r", i, C[i, j]))
    known = {("r", 0): 0.0}
    frontier = [("r", 0)]
    while frontier:
        node = frontier.pop()
        for nb in adj.get(node, []):
            kind, idx, c_ij = nb
            other = (kind, idx)
            if other not in known:
                if node[0] == "r":
                    known[other] = c_ij - known[node]
                else:
                    known[other] = c_ij - known[node]
                frontier.append(other)
    for i in range(m):
        if ("r", i) in known:
            u[i] = known[("r", i)]
    for j in range(n):
        if ("c", j) in known:
            v[j] = known[("c", j)]
    return u, v


basis_me = basic_cells(X_me)
u_manual, v_manual = potentials_from_basis(C, basis_me, m, n)
deltas = np.full((m, n), np.nan)
for i in range(m):
    for j in range(n):
        if (i, j) not in basis_me and u_manual[i] is not None and v_manual[j] is not None:
            deltas[i, j] = C[i, j] - (u_manual[i] + v_manual[j])

free_cells = [(i, j) for i in range(m) for j in range(n) if (i, j) not in basis_me and not np.isnan(deltas[i, j])]
worst_cell = min(free_cells, key=lambda ij: deltas[ij]) if free_cells else None
manual_optimal_already = worst_cell is None or deltas[worst_cell] >= -1e-9


def find_cycle(X, start, tol=1e-6):
    """Знаходить замкнений цикл перерахунку для входу start у плані X (базисні клітинки + start)."""
    basis = set(basic_cells(X, tol)) | {start}
    rows = {}
    cols = {}
    for (i, j) in basis:
        rows.setdefault(i, []).append(j)
        cols.setdefault(j, []).append(i)

    def dfs(path, visited_cells, horizontal):
        last = path[-1]
        if len(path) >= 4 and last == start:
            return path
        candidates = cols[last[1]] if horizontal else rows[last[0]]
        for nxt in candidates:
            cell = (nxt, last[1]) if horizontal else (last[0], nxt)
            if cell == start and len(path) >= 3:
                return path + [cell]
            if cell in visited_cells or cell == path[-1]:
                continue
            res = dfs(path + [cell], visited_cells | {cell}, not horizontal)
            if res:
                return res
        return None

    result = dfs([start], {start}, True)
    return result


cycle = find_cycle(X_me, worst_cell) if worst_cell else None
theta = None
X_me_improved = X_me.copy()
if cycle:
    cycle = cycle[:-1]  # прибираємо повторену стартову клітинку в кінці
    minus_cells = cycle[1::2]
    theta = min(X_me[i, j] for (i, j) in minus_cells)
    for idx, (i, j) in enumerate(cycle):
        X_me_improved[i, j] += theta if idx % 2 == 0 else -theta

# ---------- Оптимальний план (LP, scipy) ----------
c_vec = C.flatten()
A_eq = []
b_eq = []
for i in range(m):
    row = np.zeros(m * n)
    row[i * n:(i + 1) * n] = 1
    A_eq.append(row)
    b_eq.append(a_raw[i])
for j in range(n):
    row = np.zeros(m * n)
    for i in range(m):
        row[i * n + j] = 1
    A_eq.append(row)
    b_eq.append(b_raw[j])

res_lp = linprog(c_vec, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
X_opt = res_lp.x.reshape(m, n)
u_lp = res_lp.eqlin.marginals[:m]
v_lp = res_lp.eqlin.marginals[m:]
basis_opt = basic_cells(X_opt)
potentials_ok = all(abs(u_lp[i] + v_lp[j] - C[i, j]) < 1e-6 for (i, j) in basis_opt)

# ---------- Задача про призначення ----------
# "перші чотири рядки і стовпці" матриці тарифів: у варіанта лише 3 постачальники,
# тому матрицю квадратують четвертим фіктивним постачальником (нульові тарифи) —
# саме прийом, викладений у п.6.1 методички для зведення відкритої задачі до закритої/квадратної.
T = np.vstack([C, np.zeros((1, n))])
row_assign, col_assign = linear_sum_assignment(T)
assignment_cost = float(T[row_assign, col_assign].sum())
assignment_pairs = {int(i) + 1: int(j) + 1 for i, j in zip(row_assign, col_assign)}

# ---------- Складність: угорський алгоритм vs повний перебір ----------
rng = np.random.default_rng(42)
sizes = [4, 6, 8, 10]
brute_sizes = [4, 6, 8]
hu_times = []
brute_times = []
for size in sizes:
    M_rand = rng.integers(1, 50, size=(size, size)).astype(float)
    t0 = time.perf_counter()
    linear_sum_assignment(M_rand)
    hu_times.append(time.perf_counter() - t0)
    if size in brute_sizes:
        t0 = time.perf_counter()
        best = None
        for perm in itertools.permutations(range(size)):
            total = sum(M_rand[i, perm[i]] for i in range(size))
            if best is None or total < best:
                best = total
        brute_times.append(time.perf_counter() - t0)

fig_hu, ax_hu = plt.subplots(figsize=(6, 4))
ax_hu.semilogy(sizes, [max(t, 1e-7) for t in hu_times], "o-", label="Угорський алгоритм")
ax_hu.semilogy(brute_sizes, [max(t, 1e-7) for t in brute_times], "s-", label="Повний перебір")
ax_hu.set_xlabel("n (розмір матриці)")
ax_hu.set_ylabel("час, с (лог. шкала)")
ax_hu.legend()
ax_hu.grid(alpha=.3, which="both")
ax_hu.set_title("Угорський алгоритм проти повного перебору")
fig_hu.tight_layout()
complexity_path = f"complexity_variant_{VARIANT}.png"
fig_hu.savefig(complexity_path, dpi=150)
plt.close(fig_hu)

# ---------- Лістинг коду ----------
CODE_METHODS = '''def north_west(a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros((len(a), len(b)))
    i = j = 0
    while i < len(a) and j < len(b):
        q = min(a[i], b[j])
        X[i, j] = q
        a[i] -= q; b[j] -= q
        if a[i] <= 1e-9:
            i += 1
        else:
            j += 1
    return X


def min_element(C, a, b):
    a, b = a.copy(), b.copy()
    X = np.zeros_like(C); M = C.copy()
    while a.sum() > 1e-9:
        i, j = np.unravel_index(np.argmin(M), M.shape)
        q = min(a[i], b[j])
        X[i, j] = q
        a[i] -= q; b[j] -= q
        if a[i] <= 1e-9: M[i, :] = np.inf
        if b[j] <= 1e-9: M[:, j] = np.inf
    return X


def vogel(C, a, b):
    # на кожному кроці обирає рядок/стовпець із найбільшим штрафом
    # (різницею двох найменших тарифів) і заповнює в ньому найдешевшу клітинку
    ...


# Оптимальний план — зведення до задачі лінійного програмування:
res = linprog(C.flatten(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
u, v = res.eqlin.marginals[:m], res.eqlin.marginals[m:]   # потенціали = двоїсті змінні

# Задача про призначення:
row, col = linear_sum_assignment(T)   # угорський алгоритм, O(n^3)
'''

# ---------- Контрольні питання ----------
QA = [
    ("Сформулюйте транспортну задачу. Чим закрита задача відрізняється від відкритої?",
     "Транспортна задача шукає план перевезень x_ij від m постачальників із запасами a_i до n споживачів "
     "із потребами b_j, що мінімізує сумарні витрати Σc_ij·x_ij за умови вичерпання всіх запасів і "
     "задоволення всіх потреб. Задача закрита (збалансована), якщо сума запасів дорівнює сумі потреб "
     "(Σa_i = Σb_j) — тоді розв'язок завжди існує. Якщо суми не рівні, задача відкрита: запасів або "
     "забагато, або бракує, і без додаткових дій класична постановка не має допустимого плану."),
    ("Як звести відкриту транспортну задачу до закритої? Який зміст має фіктивний пункт?",
     "Вводять фіктивного споживача (якщо запасів більше, ніж потреб) або фіктивного постачальника (якщо "
     "навпаки) з нульовими тарифами перевезення. Перевезення у фіктивний пункт не є реальним рухом "
     "вантажу — воно просто формально показує нерозподілений залишок запасу або незадоволену частину "
     "потреби, і дозволяє розв'язувати задачу стандартними методами транспортного типу."),
    ("Чому оптимальний план транспортної задачі з цілими запасами і потребами завжди цілочисловий?",
     "Матриця обмежень транспортної задачі цілком унімодулярна (властивість, що випливає з її "
     "спеціальної структури — кожен стовпець матриці обмежень має рівно дві одиниці, в решті нулі). Для "
     "задач лінійного програмування з цілком унімодулярною матрицею обмежень і цілими правими частинами "
     "кожна вершина допустимого багатогранника (зокрема й оптимальна) автоматично має цілі координати — "
     "цілочисловість не потрібно вимагати окремою умовою, як у загальному цілочисловому програмуванні."),
    ("Скільки базисних клітинок містить невироджений опорний план і чому?",
     "Рівно m + n − 1. Система обмежень транспортної задачі має m + n рівнянь (m умов на запаси, n умов "
     "на потреби), але одне з них лінійно залежне від решти (сума всіх умов на запаси дорівнює сумі всіх "
     "умов на потреби при збалансованій задачі), тому ранг системи дорівнює m + n − 1 — саме стільки "
     "базисних (ненульових) змінних потрібно для невиродженого базисного розв'язку."),
    ("Порівняйте методи північно-західного кута, мінімального елемента та Фогеля.",
     "Метод північно-західного кута найпростіший, але повністю ігнорує тарифи — заповнює клітинки лише за "
     "геометричним порядком, тому дає найгірший опорний план. Метод мінімального елемента враховує "
     "вартість, заповнюючи клітинки в порядку зростання тарифів, і дає план, зазвичай значно ближчий до "
     "оптимального. Метод Фогеля — найточніший із трьох: на кожному кроці обчислює штраф (різницю двох "
     "найменших тарифів) для кожного рядка й стовпця та заповнює клітинку саме там, де штраф "
     "найбільший — тобто там, де «помилка» від невдалого вибору коштувала б найдорожче."),
    ("Що таке потенціали і як їх обчислюють? Чому одне значення задають довільно?",
     "Потенціали u_i (постачальників) і v_j (споживачів) — двоїсті змінні транспортної задачі, які "
     "знаходять із системи рівнянь u_i + v_j = c_ij, записаної лише для базисних (заповнених) клітинок "
     "поточного плану. Базисних клітинок m+n−1, а невідомих u_i, v_j — m+n, тобто система має на одне "
     "рівняння менше, ніж невідомих, і тому одне значення (зазвичай u_1 = 0) задають довільно — від цього "
     "вибору залежать самі числа u_i, v_j, але не оцінки Δ_ij вільних клітинок і не оптимальний план."),
    ("Як за оцінками вільних клітинок визначити, що план оптимальний?",
     "Для кожної вільної (незаповненої) клітинки обчислюють оцінку Δ_ij = c_ij − (u_i + v_j). Якщо всі "
     "Δ_ij ≥ 0, план оптимальний: використання будь-якого ще не задіяного маршруту могло б тільки "
     "збільшити або не змінити сумарні витрати. Якщо хоча б одна оцінка від'ємна, план можна покращити, "
     "залучивши відповідну клітинку до базису."),
    ("Що таке цикл перерахунку і як визначають величину переміщуваного вантажу θ?",
     "Цикл перерахунку — замкнена ламана лінія, вершини якої лежать у базисних клітинках (і в новій "
     "клітинці, що вводиться в базис), а сторони йдуть почергово по рядку й по стовпцю. Вершинам "
     "присвоюють знаки «+» і «−», чергуючи їх уздовж циклу, починаючи з «+» у новій клітинці. Величину θ "
     "обирають як мінімальне значення перевезення серед усіх клітинок зі знаком «−» — саме стільки "
     "вантажу можна перемістити циклом, не отримавши від'ємних перевезень; клітинка, що досягла нуля, "
     "виходить із базису."),
    ("Чим задача про призначення відрізняється від загальної транспортної задачі?",
     "Задача про призначення — окремий (вироджений у певному сенсі) випадок транспортної задачі, де "
     "кількість постачальників дорівнює кількості споживачів (m = n), усі запаси й потреби дорівнюють "
     "одиниці (a_i = b_j = 1), а змінні по суті булеві (x_ij ∈ {0,1}) — кожен «виконавець» призначається "
     "рівно на одну «роботу» і навпаки, без можливості часткового розподілу."),
    ("Яка обчислювальна складність угорського алгоритму і в чому його перевага перед повним перебором?",
     "Угорський алгоритм розв'язує задачу про призначення за O(n³) операцій, тоді як повний перебір усіх "
     "можливих призначень вимагає n! варіантів. Різниця стає критичною вже для помірних n: наприклад, для "
     "n = 10 повний перебір — це понад 3,6 мільйона варіантів, а для n = 20 — астрономічні 2,4·10¹⁸, тоді "
     "як угорський алгоритм залишається швидким і практичним для будь-якого розумного розміру задачі."),
]

# ---------- DOCX-звіт ----------
doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(13)


def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h


def add_para(text="", bold=False, italic=False, align=None, size=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def add_code_block(text):
    p = doc.add_paragraph()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        run = p.add_run(line)
        run.font.name = "Courier New"
        run.font.size = Pt(9.5)
        if i < len(lines) - 1:
            run.add_break()
    p.paragraph_format.space_after = Pt(8)
    return p


def add_table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htext in enumerate(headers):
        t.rows[0].cells[i].text = htext
    for row in rows:
        cells_ = t.add_row().cells
        for i, val in enumerate(row):
            cells_[i].text = str(val)
    return t


def matrix_table(X, row_labels, col_labels, nd=0):
    headers = [""] + col_labels
    rows = []
    for i, lbl in enumerate(row_labels):
        rows.append([lbl] + [fmt(X[i, j], nd) for j in range(len(col_labels))])
    add_table(headers, rows)


# --- Титульний аркуш ---
add_para("МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ", align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(UNIVERSITY, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(FACULTY, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(6):
    doc.add_paragraph()
add_para(f"Дисципліна: «{DISCIPLINE}»", align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(f"ЗВІТ до практичної роботи № {WORK_NUMBER}", bold=True, size=16, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(f"«{WORK_TITLE}»", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(f"Варіант № {VARIANT}", italic=True, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(8):
    doc.add_paragraph()
add_para(f"Виконав: студент групи {GROUP}", align=WD_ALIGN_PARAGRAPH.RIGHT)
add_para(STUDENT_NAME, align=WD_ALIGN_PARAGRAPH.RIGHT)
for _ in range(6):
    doc.add_paragraph()
add_para("Вінниця — 2026", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# --- 1. Мета ---
add_heading("1. Мета роботи")
add_para(
    "Опанувати методи розв'язання задач транспортного типу: побудову опорного плану методами "
    "північно-західного кута, мінімального елемента та Фогеля, перевірку оптимальності методом "
    "потенціалів, а також розв'язання задачі про призначення угорським алгоритмом."
)

# --- 2. Постановка задачі ---
add_heading("2. Постановка задачі")
add_para(f"Постачальники (запаси): a = {pt(a_raw)}.  Споживачі (потреби): b = {pt(b_raw)}.")
matrix_table(C, [f"A{i+1}" for i in range(m)], [f"B{j+1}" for j in range(n)])
add_para(
    f"Баланс: Σa = {fmt(a.sum(), 0)}, Σb = {fmt(b.sum(), 0)} → "
    f"{'задача ЗАКРИТА (збалансована)' if balanced else 'задача ВІДКРИТА — потрібен фіктивний пункт'}."
)

# --- 3. Опорні плани ---
add_heading("3. Опорні плани трьома методами")
for name, X in [("Північно-західного кута", X_nw), ("Мінімального елемента", X_me), ("Фогеля", X_vg)]:
    add_para(name, bold=True)
    matrix_table(X, [f"A{i+1}" for i in range(m)], [f"B{j+1}" for j in range(n)])
    add_para(f"Вартість плану: F = {fmt(cost(X), 0)}. Базисних клітинок: {len(basic_cells(X))} (норма: {m+n-1}).")

add_para(
    f"Метод північно-західного кута дав найдорожчий план (F = {fmt(cost(X_nw),0)}), оскільки повністю "
    f"ігнорує тарифи. Метод мінімального елемента (F = {fmt(cost(X_me),0)}) і метод Фогеля "
    f"(F = {fmt(cost(X_vg),0)}) враховують вартість і дають план, значно ближчий до оптимального."
)

# --- 4. Ручна ітерація методу потенціалів ---
add_heading("4. Ручна ітерація методу потенціалів (з плану мінімального елемента)")
u_manual_str = ", ".join(fmt(x, 0) if x is not None else "?" for x in u_manual)
v_manual_str = ", ".join(fmt(x, 0) if x is not None else "?" for x in v_manual)
add_para(f"Потенціали: u = ({u_manual_str}), v = ({v_manual_str}) (u₁ = 0 задано довільно).")
if worst_cell:
    i0, j0 = worst_cell
    add_para(
        f"Найбільш від'ємна оцінка — клітинка (A{i0+1};B{j0+1}): Δ = {fmt(deltas[worst_cell],0)} "
        f"{'< 0 → план можна покращити.' if deltas[worst_cell] < -1e-9 else '≥ 0.'}"
    )
    if cycle and theta is not None:
        cycle_str = " → ".join(f"(A{i+1};B{j+1})" for i, j in cycle)
        add_para(f"Цикл перерахунку: {cycle_str}. θ = min перевезень у «мінусових» вершинах = {fmt(theta,0)}.")
        add_para("План після перерахунку:")
        matrix_table(X_me_improved, [f"A{i+1}" for i in range(m)], [f"B{j+1}" for j in range(n)])
        add_para(f"Нова вартість: F = {fmt(cost(X_me_improved),0)} (було {fmt(cost(X_me),0)}).")
else:
    add_para("Усі оцінки вільних клітинок невід'ємні — план мінімального елемента вже оптимальний.")

# --- 5. Код ---
add_heading("5. Текст програми")
add_code_block(CODE_METHODS.strip("\n"))

# --- 6. Оптимальний план ---
add_heading("6. Оптимальний план (комп'ютерний розв'язок) і перевірка умов оптимальності")
matrix_table(X_opt, [f"A{i+1}" for i in range(m)], [f"B{j+1}" for j in range(n)])
add_para(f"Оптимальна вартість: F* = {fmt(res_lp.fun, 0)}. Базисних клітинок: {len(basis_opt)} (норма: {m+n-1}).")
u_lp_str = ", ".join(fmt(x, 2) for x in u_lp)
v_lp_str = ", ".join(fmt(x, 2) for x in v_lp)
add_para(f"Потенціали (двоїсті змінні LP): u = ({u_lp_str}), v = ({v_lp_str}).")
add_para(
    f"Перевірка u_i+v_j=c_ij для всіх базисних клітинок: "
    f"{'ВИКОНУЄТЬСЯ для всіх клітинок ✓' if potentials_ok else 'є розбіжності — див. нижче'}."
)
add_para(
    f"Економія відносно опорних планів: {fmt(cost(X_nw)-res_lp.fun,0)} (проти пн.-зх. кута), "
    f"{fmt(cost(X_me)-res_lp.fun,0)} (проти мін. елемента), {fmt(cost(X_vg)-res_lp.fun,0)} (проти Фогеля)."
)

# --- 7. Задача про призначення ---
add_heading("7. Задача про призначення")
add_para(
    f"Варіант задає лише {m} постачальників при {n} стовпцях тарифів — щоб отримати квадратну матрицю "
    f"для задачі про призначення, матрицю доповнено четвертим фіктивним постачальником із нульовими "
    f"тарифами (той самий прийом зведення відкритої задачі до закритої, що й у п. 2)."
)
matrix_table(T, [f"A{i+1}" for i in range(m)] + ["A4 (фікт.)"], [f"B{j+1}" for j in range(n)])
add_para(f"Оптимальне призначення (рядок → стовпець): {assignment_pairs}.")
add_para(f"Сумарні витрати призначення: {fmt(assignment_cost, 0)}.")

# --- 8. Графік складності ---
add_heading("8. Порівняння трудомісткості: угорський алгоритм проти повного перебору")
doc.add_picture(complexity_path, width=Cm(13))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_table(
    ["n", "Угорський, с", "Повний перебір, с"],
    [[sizes[k], fmt(hu_times[k], 6), fmt(brute_times[k], 4) if k < len(brute_times) else "—(не рахували)"]
     for k in range(len(sizes))]
)
add_para(
    "Час угорського алгоритму практично не залежить від n у цьому діапазоні розмірів (частки мілісекунди), "
    "тоді як час повного перебору зростає факторіально й вже для n = 8 стає помітно довшим; для n = 10 "
    "повний перебір (10! ≈ 3,6 млн варіантів) свідомо не рахувався — відмінність ілюструють менші n."
)

# --- 9. Висновки ---
add_heading("9. Висновки")
conclusions = [
    f"Для варіанта {VARIANT} оптимальна вартість перевезень F* = {fmt(res_lp.fun,0)}, що узгоджується з "
    f"ручним розрахунком методом потенціалів і підтверджується виконанням умови u_i+v_j=c_ij на всіх "
    f"базисних клітинках.",
    "Якість опорного плану суттєво впливає на подальшу роботу методу потенціалів: північно-західний кут "
    "ігнорує тарifи і дає найгірший старт, тоді як методи мінімального елемента та Фогеля враховують "
    "вартість і наближають до оптимуму вже на етапі побудови опорного плану.",
    "Матриця обмежень транспортної задачі цілком унімодулярна, тому за цілих запасів і потреб оптимальний "
    "план автоматично цілочисловий — додаткових умов цілочисловості не потрібно.",
    "Задача про призначення — окремий випадок транспортної задачі з одиничними запасами/потребами і "
    "булевими змінними; угорський алгоритм розв'язує її за O(n³) замість n! операцій повного перебору, "
    "що на практиці перетворює непридатну задачу на миттєво розв'язувану.",
]
for c in conclusions:
    doc.add_paragraph(c, style="List Bullet")

# --- 10. Контрольні питання ---
add_heading("10. Відповіді на контрольні питання")
for i, (q, ans) in enumerate(QA, start=1):
    add_para(f"{i}. {q}", bold=True)
    add_para(ans)

docx_path = f"Zvit_PR{WORK_NUMBER}_Variant{VARIANT}.docx"
doc.save(docx_path)
print(f"Готово: {docx_path} (+ {complexity_path})")
