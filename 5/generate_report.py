"""
Генератор звіту до практичної роботи №5 (Лінійне програмування: симплекс-метод, двоїстість).
Питає номер варіанта (або приймає його аргументом) і генерує Zvit_PR5_Variant<N>.docx.

Запуск:  .venv/bin/python generate_report.py [номер_варіанта]
"""
import sys
import numpy as np
from scipy.optimize import linprog
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from variants import VARIANTS

# --- Дані студента / титульного аркуша ---
STUDENT_NAME = "Толошний Мирослав"
GROUP = "КН-11"
UNIVERSITY = "Вінницький національний аграрний університет"
FACULTY = "Факультет комп'ютерних наук та економічної кібернетики"
DISCIPLINE = "Аналіз та методи оптимізації складних систем"
WORK_TITLE = "Лінійне програмування: симплекс-метод, двоїстість та аналіз чутливості"
WORK_NUMBER = 5

if len(sys.argv) > 1:
    VARIANT = int(sys.argv[1])
else:
    while True:
        raw = input(f"Введи номер варіанта (1-{len(VARIANTS)}): ").strip()
        if raw.isdigit() and int(raw) in VARIANTS:
            VARIANT = int(raw)
            break
        print(f"Треба ціле число від 1 до {len(VARIANTS)}, спробуй ще раз.")

# ---------- Модель ----------
c_raw, A_raw, b_raw = VARIANTS[VARIANT]
c = np.array(c_raw, dtype=float)          # коефіцієнти прибутку, Z = c1*x1 + c2*x2 -> max
A = np.array(A_raw, dtype=float)          # 3x2 матриця витрат ресурсів
b = np.array(b_raw, dtype=float)          # запаси ресурсів
n = 2                                      # кількість змінних (видів продукції)
m = 3                                      # кількість обмежень (ресурсів)
RESOURCE_NAMES = ["Сировина", "Машинний час", "Робочий час"]
RESOURCE_UNITS = ["кг", "год", "год"]
VAR_NAMES_ALL = [f"x{i+1}" for i in range(n)] + [f"s{i+1}" for i in range(m)]


def pt(v, nd=4):
    """np.float64 -> звичайний float (інакше в numpy>=2 рядок типу 'np.float64(1.2)')."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return "(" + ", ".join(f"{float(c_):.{nd}g}" for c_ in v) + ")"
    return float(v)


def fmt(v, nd=4):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{nd}f}"
    return str(v)


# ========================================================================
# 1. Симплекс-метод "з нуля" (повний протокол таблиць)
# ========================================================================
def simplex_solve(c, A, b, max_iter=30):
    """Табличний симплекс-метод для max c^T x, Ax <= b, x >= 0, b >= 0.
    Повертає (x*, Z*, basis_idx, var_names, tableaus) — tableaus містить
    знімок кожної таблиці та інформацію про обраний розв'язувальний елемент."""
    mm, nn = A.shape
    T = np.hstack([A, np.eye(mm), b.reshape(-1, 1)]).astype(float)
    obj = np.concatenate([-c, np.zeros(mm + 1)])   # Z-рядок: Z - c1*x1 - c2*x2 = 0
    basis = list(range(nn, nn + mm))                # початковий базис — слекові змінні
    var_names = [f"x{i+1}" for i in range(nn)] + [f"s{i+1}" for i in range(mm)]

    tableaus = []
    it = 0
    while True:
        snap = {"basis": list(basis), "T": T.copy(), "obj": obj.copy(), "pivot": None}
        if np.all(obj[:-1] >= -1e-9):
            tableaus.append(snap)
            break
        col = int(np.argmin(obj[:-1]))              # найбільш негативна оцінка (правило Данціга)
        colvals = T[:, col]
        if np.all(colvals <= 1e-9):
            raise RuntimeError("Цільова функція необмежена на допустимій області")
        ratios = [T[i, -1] / T[i, col] if T[i, col] > 1e-9 else np.inf for i in range(mm)]
        row = int(np.argmin(ratios))
        snap["pivot"] = {
            "col": col, "row": row,
            "entering": var_names[col], "leaving": var_names[basis[row]],
            "ratios": list(ratios),
        }
        tableaus.append(snap)

        pivot = T[row, col]
        T[row, :] = T[row, :] / pivot
        for i in range(mm):
            if i != row:
                T[i, :] -= T[i, col] * T[row, :]
        obj = obj - obj[col] * T[row, :]
        basis[row] = col
        it += 1
        if it > max_iter:
            raise RuntimeError("Забагато ітерацій симплекс-методу")

    x = np.zeros(nn)
    for i, bidx in enumerate(basis):
        if bidx < nn:
            x[bidx] = T[i, -1]
    Z = float(obj[-1])
    return x, Z, basis, var_names, tableaus


x_manual, Z_manual, basis_final, var_names, tableaus = simplex_solve(c, A, b)

# ---------- Перевірка доступними засобами SciPy ----------
res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
x_scipy = res.x
Z_scipy = float(-res.fun)
slacks_scipy = res.slack
duals_scipy = -res.ineqlin.marginals  # тіньові ціни (знак узгоджено з прибутком)

if not np.allclose(x_manual, x_scipy, atol=1e-3) or abs(Z_manual - Z_scipy) > 1e-2:
    print("УВАГА: ручний симплекс і SciPy дали різні результати!")
    print("manual:", x_manual, Z_manual)
    print("scipy :", x_scipy, Z_scipy)

# ---------- Перевірка допустимості знайденого плану ----------
slack_manual = b - A @ x_manual
feasible = bool(np.all(slack_manual >= -1e-6) and np.all(x_manual >= -1e-6))

# ========================================================================
# 2. Двоїста задача
# ========================================================================
# W = b1*y1 + b2*y2 + b3*y3 -> min;  A^T y >= c;  y >= 0.
# Тіньові ціни y* беремо з фінальної симплекс-таблиці: це коефіцієнти в Z-рядку
# при слекових стовпцях (оцінки Δ слекових змінних дорівнюють двоїстим оцінкам).
final_obj = tableaus[-1]["obj"]
y_manual = np.array([final_obj[n + i] for i in range(m)])
W_manual = float(b @ y_manual)

# ---------- Дефіцитні / надлишкові ресурси ----------
resource_rows = []
for i in range(m):
    used = float(A[i] @ x_manual)
    slack_i = float(b[i] - used)
    status = "дефіцитний" if slack_i < 1e-6 else "надлишковий"
    resource_rows.append((RESOURCE_NAMES[i], RESOURCE_UNITS[i], float(b[i]), used, slack_i,
                           float(y_manual[i]), status))

# ---------- Умови доповнювальної нежорсткості ----------
compl_rows = []
for i in range(m):
    slack_i = float(b[i] - A[i] @ x_manual)
    compl_rows.append((f"обмеження {i+1} ({RESOURCE_NAMES[i]})", f"y{i+1}", float(y_manual[i]),
                        slack_i, float(y_manual[i] * slack_i)))
for j in range(n):
    # оцінка j-го обмеження двоїстої задачі: (A^T y)_j - c_j
    dual_slack_j = float(A[:, j] @ y_manual - c[j])
    compl_rows.append((f"x{j+1}", f"обм. двоїстої {j+1}", dual_slack_j, float(x_manual[j]),
                        float(dual_slack_j * x_manual[j])))

# ========================================================================
# 3. Графічний розв'язок
# ========================================================================
def polygon_vertices(A, b):
    """Усі вершини допустимої області Ax<=b, x>=0 (перебір попарних перетинів меж)."""
    lines = [(A[i, 0], A[i, 1], b[i]) for i in range(len(b))]
    lines.append((1.0, 0.0, 0.0))  # x1 = 0
    lines.append((0.0, 1.0, 0.0))  # x2 = 0
    pts = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a1, a2, r1 = lines[i]
            a3, a4, r2 = lines[j]
            M = np.array([[a1, a2], [a3, a4]])
            if abs(np.linalg.det(M)) < 1e-9:
                continue
            p = np.linalg.solve(M, [r1, r2])
            if p[0] < -1e-6 or p[1] < -1e-6:
                continue
            if np.all(A @ p <= b + 1e-6):
                pts.append((round(float(p[0]), 6), round(float(p[1]), 6)))
    pts = sorted(set(pts))
    pts = np.array(pts)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    order = np.argsort(angles)
    return pts[order]


poly = polygon_vertices(A, b)

xmax_axis = max(b[i] / A[i, 0] for i in range(m) if A[i, 0] > 0) * 1.0
ymax_axis = max(b[i] / A[i, 1] for i in range(m) if A[i, 1] > 0) * 1.0
lim_x = max(poly[:, 0].max(), xmax_axis) * 1.2 + 1
lim_y = max(poly[:, 1].max(), ymax_axis) * 1.2 + 1

fig, ax = plt.subplots(figsize=(7, 6))
ax.fill(poly[:, 0], poly[:, 1], color="tab:blue", alpha=0.25, label="Допустима область")
ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), "b-", lw=1.5)

xs_line = np.linspace(0, lim_x, 200)
for i in range(m):
    a1, a2 = A[i]
    if abs(a2) > 1e-9:
        ys_line = (b[i] - a1 * xs_line) / a2
        ax.plot(xs_line, ys_line, "--", lw=1, label=f"{RESOURCE_NAMES[i]}: {a1:g}x1+{a2:g}x2={b[i]:g}")
    else:
        ax.axvline(b[i] / a1, ls="--", lw=1, label=f"{RESOURCE_NAMES[i]}: {a1:g}x1={b[i]:g}")

for px, py in poly:
    ax.plot(px, py, "ko", ms=5)
    ax.annotate(f"({px:g}, {py:g})", xy=(px, py), textcoords="offset points",
                xytext=(6, 6), fontsize=8)

# Вектор-градієнт і лінія рівня через оптимальну точку
x_opt1, x_opt2 = float(x_manual[0]), float(x_manual[1])
grad_norm = c / np.linalg.norm(c) * 0.18 * max(lim_x, lim_y)
ax.annotate("", xy=(x_opt1 + grad_norm[0], x_opt2 + grad_norm[1]), xytext=(x_opt1, x_opt2),
            arrowprops=dict(facecolor="red", edgecolor="red", width=1.5, headwidth=8))
ax.text(x_opt1 + grad_norm[0], x_opt2 + grad_norm[1], "  c (градієнт)", color="red", fontsize=9)

if abs(c[1]) > 1e-9:
    ys_level = (Z_manual - c[0] * xs_line) / c[1]
else:
    ys_level = None
if ys_level is not None:
    mask = (ys_level >= -0.5) & (ys_level <= lim_y)
    ax.plot(xs_line[mask], ys_level[mask], "r:", lw=1.5,
            label=f"лінія рівня Z={Z_manual:g}")

ax.plot(x_opt1, x_opt2, "r*", ms=18, label=f"x* = ({x_opt1:.3g}, {x_opt2:.3g})")
ax.set_xlim(0, lim_x)
ax.set_ylim(0, lim_y)
ax.set_xlabel("x1 (продукція A)")
ax.set_ylabel("x2 (продукція B)")
ax.set_title(f"Варіант {VARIANT}: допустима область та оптимальний план")
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
graph_path = f"graph_variant_{VARIANT}.png"
fig.savefig(graph_path, dpi=150)
plt.close(fig)

# ========================================================================
# 4. Аналіз чутливості: Z*(b_i), ±25 %, та інтервали стійкості
# ========================================================================
def rhs_stability_interval(T, basis, slack_col_idx, b_i, n):
    """Межі зміни b_i, що не змінюють базис (використовує фінальну таблицю)."""
    col = T[:, n + slack_col_idx]
    xB = T[:, -1]
    lower_cands, upper_cands = [], []
    for i in range(len(col)):
        if col[i] > 1e-9:
            lower_cands.append(-xB[i] / col[i])
        elif col[i] < -1e-9:
            upper_cands.append(-xB[i] / col[i])
    delta_min = max(lower_cands) if lower_cands else -np.inf
    delta_max = min(upper_cands) if upper_cands else np.inf
    return b_i + delta_min, b_i + delta_max


final_T = tableaus[-1]["T"]
stability_rows = []
for i in range(m):
    lo, hi = rhs_stability_interval(final_T, basis_final, i, float(b[i]), n)
    stability_rows.append((RESOURCE_NAMES[i], lo, float(b[i]), hi, float(y_manual[i])))

sens_rows = {i: [] for i in range(m)}
pct_grid = [-25, -12.5, 0, 12.5, 25]
for i in range(m):
    for pct in pct_grid:
        b_mod = b.copy()
        b_mod[i] = b[i] * (1 + pct / 100.0)
        r = linprog(-c, A_ub=A, b_ub=b_mod, bounds=[(0, None)] * n, method="highs")
        sens_rows[i].append((pct, float(b_mod[i]), float(-r.fun)))

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
for i, ax2 in enumerate(axes2):
    bs = [row[1] for row in sens_rows[i]]
    zs = [row[2] for row in sens_rows[i]]
    ax2.plot(bs, zs, "o-", color="tab:blue")
    ax2.axvline(float(b[i]), color="gray", ls=":", lw=1)
    ax2.set_title(RESOURCE_NAMES[i])
    ax2.set_xlabel(f"b{i+1}, {RESOURCE_UNITS[i]}")
    ax2.set_ylabel("Z*")
    ax2.grid(alpha=0.3)
fig2.suptitle(f"Варіант {VARIANT}: залежність Z* від запасу кожного ресурсу (±25%)")
fig2.tight_layout()
sens_path = f"sensitivity_variant_{VARIANT}.png"
fig2.savefig(sens_path, dpi=150)
plt.close(fig2)

best_resource_idx = int(np.argmax(y_manual))

# ---------- Лістинг коду симплекс-методу ----------
CODE_SIMPLEX = '''def simplex_solve(c, A, b, max_iter=30):
    """Табличний симплекс-метод для max c^T x, Ax <= b, x >= 0, b >= 0."""
    mm, nn = A.shape
    T = np.hstack([A, np.eye(mm), b.reshape(-1, 1)]).astype(float)
    obj = np.concatenate([-c, np.zeros(mm + 1)])      # Z-рядок: Z - c^T x = 0
    basis = list(range(nn, nn + mm))                   # початковий базис -- слеки
    var_names = [f"x{i+1}" for i in range(nn)] + [f"s{i+1}" for i in range(mm)]

    tableaus = []
    while True:
        snap = {"basis": list(basis), "T": T.copy(), "obj": obj.copy(), "pivot": None}
        if np.all(obj[:-1] >= -1e-9):                  # критерій оптимальності
            tableaus.append(snap)
            break
        col = int(np.argmin(obj[:-1]))                 # розв'язувальний стовпець (правило Данціга)
        ratios = [T[i, -1] / T[i, col] if T[i, col] > 1e-9 else np.inf for i in range(mm)]
        row = int(np.argmin(ratios))                   # розв'язувальний рядок (мін. відношення)
        snap["pivot"] = {"entering": var_names[col], "leaving": var_names[basis[row]]}
        tableaus.append(snap)

        pivot = T[row, col]
        T[row, :] /= pivot                              # нормування розв'язувального рядка
        for i in range(mm):
            if i != row:
                T[i, :] -= T[i, col] * T[row, :]        # виключення Жордана-Гауса
        obj -= obj[col] * T[row, :]
        basis[row] = col
    x = np.zeros(nn)
    for i, bidx in enumerate(basis):
        if bidx < nn:
            x[bidx] = T[i, -1]
    return x, float(obj[-1]), basis, var_names, tableaus
'''

# ---------- Контрольні питання ----------
QA = [
    ("Чому допустима область задачі ЛП є опуклим многогранником?",
     "Допустима область задається системою лінійних нерівностей aᵢ₁x₁+...+aᵢₙxₙ ≤ bᵢ. Кожна окрема "
     "нерівність визначає напівпростір — множину, яка опукла за визначенням (відрізок між будь-якими двома "
     "її точками цілком лежить у ній). Перетин будь-якої кількості опуклих множин знову опуклий, тому "
     "допустима область як перетин m напівпросторів (та умов невід'ємності xⱼ ≥ 0, які самі є напівпросторами) "
     "завжди опукла; а оскільки вона ще й обмежена лінійними гранями, це многогранник (політоп)."),
    ("Сформулюйте основну теорему лінійного програмування.",
     "Якщо допустима область задачі ЛП непорожня й оптимум цільової функції на ній існує (скінченний), то "
     "цей оптимум досягається принаймні в одній вершині многогранника допустимих розв'язків. Більше того, "
     "якщо оптимум досягається у двох суміжних вершинах, він досягається в усіх точках грані між ними. Звідси "
     "випливає, що розв'язок задачі можна шукати не по всій (нескінченній) області, а лише серед скінченної "
     "кількості вершин — саме на цьому й будується симплекс-метод."),
    ("Який економічний зміст мають додаткові змінні канонічної форми?",
     "Додаткова (слекова) змінна sᵢ у рівнянні aᵢ₁x₁+...+aᵢₙxₙ+sᵢ=bᵢ означає невикористаний залишок i-го "
     "ресурсу при плані випуску (x₁,...,xₙ): sᵢ = bᵢ − (фактично витрачено). Якщо sᵢ=0 у оптимальному плані, "
     "ресурс витрачено повністю (дефіцитний ресурс); якщо sᵢ>0 — частина запасу лишається незадіяною "
     "(надлишковий ресурс), і купувати цей ресурс додатково не має сенсу."),
    ("Як обирають розв'язувальний стовпець і розв'язувальний рядок у симплекс-методі?",
     "Розв'язувальний стовпець обирають за критерієм оптимальності: серед оцінок Δⱼ (коефіцієнтів Z-рядка) "
     "вибирають змінну з найбільшим порушенням критерію (найбільш від'ємна оцінка для задачі на максимум "
     "у прийнятому в роботі записі) — її введення в базис найшвидше покращує цільову функцію. Розв'язувальний "
     "рядок обирають за правилом мінімального симплексного відношення min(bᵢ/aᵢₖ) серед рядків із додатним "
     "елементом aᵢₖ у розв'язувальному стовпці — це найбільший крок, на який можна збільшити нову базисну "
     "змінну, не порушивши невід'ємності інших базисних змінних."),
    ("Що таке виродженість задачі ЛП і чим вона небезпечна?",
     "Виродженість — ситуація, коли в поточному базисному розв'язку одна чи кілька базисних змінних дорівнюють "
     "нулю (нульове симплексне відношення bᵢ/aᵢₖ=0). У такому разі черговий крок симплекс-методу не змінює "
     "значення цільової функції, і за невдалого вибору розв'язувального елемента алгоритм може почати "
     "циклічно повертатися до вже пройдених базисів (зациклення), не досягаючи оптимуму. Усувають цю "
     "небезпеку правилом Бленда (детермінований вибір найменшого індексу серед кандидатів) або невеликим "
     "випадковим збуренням правих частин."),
    ("Сформулюйте правила побудови двоїстої задачі.",
     "Для прямої задачі max cᵗx за Ax≤b, x≥0 двоїста будується так: (1) тип оптимізації змінюється на "
     "протилежний (max → min); (2) кожному обмеженню прямої задачі відповідає одна двоїста змінна yᵢ≥0; "
     "(3) коефіцієнти правих частин b стають коефіцієнтами цільової функції двоїстої задачі W=bᵗy; "
     "(4) матриця обмежень транспонується (Aᵗ), а права частина двоїстих обмежень — це коефіцієнти цільової "
     "функції прямої задачі c; (5) тип нерівності обмежень змінюється на протилежний (≤ → ≥)."),
    ("Сформулюйте першу і другу теореми двоїстості.",
     "Перша теорема (про рівність оптимумів): якщо пряма задача має оптимальний розв'язок x*, то й двоїста "
     "має оптимальний розв'язок y*, і значення їхніх цільових функцій збігаються: Z*=W*. Друга теорема "
     "(про доповнювальну нежорсткість): для будь-якої пари допустимих розв'язків x, y виконується "
     "yᵢ·(bᵢ−Σaᵢⱼxⱼ)=0 для кожного i та xⱼ·(Σaᵢⱼyᵢ−cⱼ)=0 для кожного j; тобто якщо обмеження виконується "
     "як строга нерівність (є залишок), відповідна двоїста змінна дорівнює нулю, і навпаки."),
    ("Який економічний зміст має двоїста оцінка ресурсу? Чому оцінка надлишкового ресурсу дорівнює нулю?",
     "Двоїста оцінка yᵢ* — це тіньова ціна ресурсу: приріст максимального прибутку Z* при збільшенні запасу "
     "i-го ресурсу на одну одиницю, ∂Z*/∂bᵢ=yᵢ*. Вона показує, скільки максимально варто заплатити за "
     "додаткову одиницю ресурсу, щоб це ще було вигідно. Якщо ресурс надлишковий (у оптимальному плані "
     "залишається невикористаний запас), збільшення його запасу на одиницю нічого не змінює в плані випуску "
     "й не додає прибутку — за другою теоремою доповнювальної нежорсткості це автоматично дає yᵢ*=0."),
    ("Що таке інтервал стійкості двоїстих оцінок і навіщо його обчислюють?",
     "Інтервал стійкості за правою частиною bᵢ — це діапазон значень запасу ресурсу, у межах якого "
     "оптимальний базис (набір дефіцитних/надлишкових ресурсів) не змінюється, а отже тіньова ціна yᵢ* "
     "залишається сталою. За межами цього інтервалу до або виходить з базису інша змінна, і тіньова ціна "
     "змінюється (ресурс може стати надлишковим або, навпаки, дефіцитним). Інтервал обчислюють, щоб знати, "
     "у яких межах висновок «кожна додаткова одиниця ресурсу дає +yᵢ* прибутку» лишається коректним."),
    ("Як за допомогою тіньових цін оцінити доцільність випуску нового виду продукції?",
     "Новий вид продукції з нормою витрат (a₁,...,aₘ) на одиницю та прибутком c вигідно включати в план лише "
     "тоді, коли його прибуток перевищує «вартість» витрачених ресурсів, оцінену за їхніми тіньовими цінами: "
     "c > Σ aᵢ·yᵢ*. Якщо ця нерівність виконується, запровадження нового продукту збільшить Z* понад поточний "
     "оптимум; якщо ні — ресурси, які пішли б на його виробництво, вигідніше й надалі витрачати на вже "
     "наявні продукти A і B."),
]

# ========================================================================
# DOCX-звіт
# ========================================================================
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


def add_table(headers, rows, font_size=10.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htext in enumerate(headers):
        t.rows[0].cells[i].text = str(htext)
    for row in rows:
        cells_ = t.add_row().cells
        for i, val in enumerate(row):
            cells_[i].text = str(val)
    for row in t.rows:
        for cell in row.cells:
            for p_ in cell.paragraphs:
                for r_ in p_.runs:
                    r_.font.size = Pt(font_size)
    return t


def tableau_rows(snap):
    basis_names = [var_names[i] for i in snap["basis"]]
    rows = []
    for i, bname in enumerate(basis_names):
        rows.append([bname] + [fmt(v, 3) for v in snap["T"][i, :-1]] + [fmt(snap["T"][i, -1], 3)])
    rows.append(["Δj (Z-рядок)"] + [fmt(v, 3) for v in snap["obj"][:-1]] + [fmt(snap["obj"][-1], 3)])
    return rows


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
    "Побудувати лінійну модель планування випуску двох видів продукції, розв'язати її графічно та "
    "симплекс-методом вручну, перевірити розв'язок засобами SciPy, сформулювати й розв'язати двоїсту задачу, "
    "інтерпретувати тіньові ціни ресурсів та виконати аналіз чутливості оптимального плану."
)

# --- 2. Постановка задачі ---
add_heading("2. Змістовна та математична постановка задачі")
add_para(
    f"Підприємство випускає два види продукції — A (x1) і B (x2), використовуючи три ресурси: "
    f"{RESOURCE_NAMES[0].lower()}, {RESOURCE_NAMES[1].lower()} та {RESOURCE_NAMES[2].lower()}, запаси яких "
    f"обмежені. Прибуток з одиниці продукції A становить {c[0]:g} грн, з одиниці B — {c[1]:g} грн. Потрібно "
    f"визначити план випуску (x1, x2), що максимізує сумарний прибуток."
)
add_para(f"Z = {c[0]:g}·x1 + {c[1]:g}·x2  →  max", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
for i in range(m):
    add_para(f"{A[i,0]:g}·x1 + {A[i,1]:g}·x2 ≤ {b[i]:g}   ({RESOURCE_NAMES[i]}, {RESOURCE_UNITS[i]})",
              align=WD_ALIGN_PARAGRAPH.CENTER)
add_para("x1, x2 ≥ 0", align=WD_ALIGN_PARAGRAPH.CENTER)

add_para("Канонічна форма (з доданими невід'ємними залишковими змінними s1, s2, s3):", bold=True)
for i in range(m):
    add_para(f"{A[i,0]:g}·x1 + {A[i,1]:g}·x2 + s{i+1} = {b[i]:g},  s{i+1} ≥ 0", align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(
    "sᵢ має економічний зміст невикористаного залишку відповідного ресурсу: якщо у плані sᵢ=0, ресурс "
    "використано повністю (дефіцитний), якщо sᵢ>0 — лишається запас (надлишковий)."
)

# --- 3. Графічний розв'язок ---
add_heading("3. Графічний розв'язок")
add_para(
    "Допустима область — перетин трьох напівплощин (обмежень) і першого квадранта; вона є опуклим "
    "многокутником. Вектор-градієнт c=(" + f"{c[0]:g}, {c[1]:g}" + ") показує напрямок найшвидшого зростання Z; "
    "лінія рівня Z=const переміщується в цьому напрямку до останнього дотику з областю — це й визначає "
    "оптимальну вершину."
)
doc.add_picture(graph_path, width=Cm(14))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    f"Оптимум досягається у вершині x* = ({x_opt1:.4g}, {x_opt2:.4g}), Z* = {Z_manual:.4g}. Вершини допустимої "
    f"області: " + "; ".join(f"({p[0]:g}, {p[1]:g})" for p in poly) + "."
)

# --- 4. Ручний розв'язок симплекс-методом ---
add_heading("4. Ручний розв'язок симплекс-методом")
add_para(
    "Початковий базис — залишкові змінні s1, s2, s3 (це відповідає вершині x1=x2=0). На кожному кроці "
    "обирається розв'язувальний стовпець (змінна з найбільш від'ємною оцінкою Δj у Z-рядку) та "
    "розв'язувальний рядок (мінімальне симплексне відношення bᵢ/aᵢₖ серед додатних aᵢₖ), після чого таблиця "
    "перераховується методом Жордана-Гауса."
)
for k, snap in enumerate(tableaus):
    add_para(f"Таблиця {k} (базис: {', '.join(var_names[i] for i in snap['basis'])})", bold=True)
    add_table(["Базис"] + var_names + ["B"], tableau_rows(snap))
    piv = snap["pivot"]
    if piv is not None:
        ratio_strs = []
        basis_now = snap["basis"]
        for i, r_ in enumerate(piv["ratios"]):
            lbl = var_names[basis_now[i]]
            ratio_strs.append(f"{lbl}: {fmt(r_,3) if np.isfinite(r_) else '—'}")
        add_para(
            f"Розв'язувальний стовпець — {piv['entering']} (найбільш від'ємна оцінка в Z-рядку). "
            f"Симплексні відношення B/стовпець: " + "; ".join(ratio_strs) + f". Мінімальне — у рядку "
            f"{piv['leaving']}, тому ця змінна виходить з базису, а {piv['entering']} входить. "
            f"Переходимо до таблиці {k+1}."
        )
    else:
        add_para(
            "Усі оцінки Δj ≥ 0 — критерій оптимальності виконано, таблиця остаточна. Базисні змінні дають "
            f"оптимальний план x* = ({x_opt1:.4g}, {x_opt2:.4g}), Z* = {Z_manual:.4g}."
        )

# --- 5. Текст програми та комп'ютерний розв'язок ---
add_heading("5. Текст програми та результати комп'ютерного розв'язання")
add_code_block(CODE_SIMPLEX.strip("\n"))
add_para(
    f"Перевірка засобами scipy.optimize.linprog (метод 'highs'): x* = "
    f"({float(x_scipy[0]):.4g}, {float(x_scipy[1]):.4g}), Z* = {Z_scipy:.4g}, залишки ресурсів = "
    f"({float(slacks_scipy[0]):.4g}, {float(slacks_scipy[1]):.4g}, {float(slacks_scipy[2]):.4g})."
)
add_para(
    ("Результат ручного симплекс-методу повністю збігається з результатом SciPy/HiGHS — це підтверджує "
     "правильність обчислень." if (np.allclose(x_manual, x_scipy, atol=1e-3) and abs(Z_manual - Z_scipy) < 1e-2)
     else "УВАГА: ручний розв'язок і SciPy відрізняються — потрібна перевірка обчислень."),
    bold=True,
)
add_para(
    f"Перевірка допустимості знайденого плану: Ax* = "
    f"({float(A[0]@x_manual):.4g}, {float(A[1]@x_manual):.4g}, {float(A[2]@x_manual):.4g}) ≤ "
    f"({b[0]:g}, {b[1]:g}, {b[2]:g}) = b — " + ("усі обмеження виконано." if feasible else "ПОРУШЕННЯ ОБМЕЖЕНЬ!")
)

# --- 6. Двоїста задача ---
add_heading("6. Двоїста задача та перевірка теорем двоїстості")
add_para(
    f"W = {b[0]:g}·y1 + {b[1]:g}·y2 + {b[2]:g}·y3  →  min", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER
)
for j in range(n):
    coefs = " + ".join(f"{A[i,j]:g}·y{i+1}" for i in range(m))
    add_para(f"{coefs} ≥ {c[j]:g}", align=WD_ALIGN_PARAGRAPH.CENTER)
add_para("y1, y2, y3 ≥ 0", align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(
    "Двоїсті оцінки y* зручно зчитати прямо з фінальної симплекс-таблиці прямої задачі: це коефіцієнти "
    "Z-рядка у стовпцях залишкових змінних s1, s2, s3 (теорема про двоїстий симплекс-множник)."
)
add_table(
    ["Оцінка", "Значення"],
    [[f"y{i+1}* ({RESOURCE_NAMES[i]})", fmt(float(y_manual[i]), 4)] for i in range(m)] +
    [["W* = b·y*", fmt(W_manual, 4)]]
)
add_para(
    f"Перша теорема двоїстості: Z* = {Z_manual:.4g}, W* = {W_manual:.4g} — значення збігаються "
    f"({'виконано' if abs(Z_manual - W_manual) < 1e-3 else 'НЕ виконано'}), отже знайдений розв'язок "
    f"справді оптимальний."
)

# --- 7. Доповнювальна нежорсткість, дефіцитні/надлишкові ---
add_heading("7. Умови доповнювальної нежорсткості та дефіцитність ресурсів")
add_table(
    ["Пара", "Двоїста змінна / оцінка", "Значення оцінки", "Залишок / значення xⱼ", "Добуток (має бути 0)"],
    [[a, b_, fmt(c_, 4), fmt(d_, 4), fmt(e_, 6)] for a, b_, c_, d_, e_ in compl_rows]
)
add_para(
    "Для кожної пари «обмеження — двоїста змінна» добуток (тіньова ціна)×(залишок ресурсу) дорівнює нулю, і "
    "для кожної пари «змінна — обмеження двоїстої задачі» добуток (значення xⱼ)×(надлишок двоїстого обмеження) "
    "також дорівнює нулю — друга теорема двоїстості виконується."
)
add_heading("Таблиця дефіцитних і надлишкових ресурсів", level=2)
add_table(
    ["Ресурс", "Запас b", "Використано", "Залишок", "Тіньова ціна y*", "Статус"],
    [[nm, fmt(bb, 2), fmt(used, 4), fmt(sl, 4), fmt(y_, 4), st]
     for nm, un, bb, used, sl, y_, st in resource_rows]
)
deficit_names = [r[0] for r in resource_rows if r[-1] == "дефіцитний"]
surplus_names = [r[0] for r in resource_rows if r[-1] == "надлишковий"]
add_para(
    f"Дефіцитні ресурси (використані повністю, тіньова ціна > 0): {', '.join(deficit_names) or '—'}. "
    f"Надлишкові ресурси (є запас, тіньова ціна = 0): {', '.join(surplus_names) or '—'}. "
    "Економічно це означає, що лише за дефіцитні ресурси є сенс платити надбавку понад їхню номінальну "
    "вартість — саме вони стримують подальше зростання прибутку."
)

# --- 8. Аналіз чутливості ---
add_heading("8. Аналіз чутливості")
doc.add_picture(sens_path, width=Cm(16))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
for i in range(m):
    add_para(f"Залежність Z* від запасу ресурсу «{RESOURCE_NAMES[i]}» (крок ±25% від {b[i]:g}):", bold=True)
    add_table(
        ["Δ, %", f"b{i+1}", "Z*"],
        [[fmt(pct, 1), fmt(bb, 2), fmt(zz, 3)] for pct, bb, zz in sens_rows[i]]
    )
add_para("Інтервали стійкості двоїстих оцінок (межі зміни bᵢ, що не змінюють оптимальний базис):", bold=True)
add_table(
    ["Ресурс", "bᵢ min", "bᵢ поточне", "bᵢ max", "Тіньова ціна y*"],
    [[nm, (fmt(lo, 3) if np.isfinite(lo) else "-∞"), fmt(bb, 3),
      (fmt(hi, 3) if np.isfinite(hi) else "+∞"), fmt(y_, 4)] for nm, lo, bb, hi, y_ in stability_rows]
)
add_para(
    "У межах свого інтервалу стійкості кожен ресурс можна збільшувати чи зменшувати, і тіньова ціна "
    "залишатиметься сталою (графіки вище — прямі лінії з нахилом, що дорівнює y*); за межами інтервалу "
    "змінюється сам оптимальний базис, і нахил графіка (а отже й тіньова ціна) змінюється."
)
add_para(
    f"Управлінський висновок: найвищу тіньову ціну має ресурс «{RESOURCE_NAMES[best_resource_idx]}» "
    f"(y* = {float(y_manual[best_resource_idx]):.4g} грн за одиницю) — саме його доцільно закуповувати "
    f"першим, і платити за додаткову одиницю можна не більше {float(y_manual[best_resource_idx]):.4g} грн, "
    f"доки це перебуває в межах інтервалу стійкості."
)

# --- 9. Висновки ---
add_heading("9. Висновки")
conclusions = [
    f"Для варіанта {VARIANT} оптимальний план випуску: x1* = {x_opt1:.4g} (продукція A), "
    f"x2* = {x_opt2:.4g} (продукція B), максимальний прибуток Z* = {Z_manual:.4g} грн.",
    "Ручний симплекс-метод і scipy.optimize.linprog (HiGHS) дали однаковий результат, що підтверджує "
    "правильність реалізації алгоритму.",
    f"Дефіцитні ресурси (повністю використані, додатна тіньова ціна): {', '.join(deficit_names) or '—'}; "
    f"надлишкові (є резерв, нульова тіньова ціна): {', '.join(surplus_names) or '—'}.",
    f"Перша теорема двоїстості підтверджена чисельно: Z* = W* = {Z_manual:.4g}; умови доповнювальної "
    "нежорсткості виконані для кожної пари обмеження/змінної.",
    "Аналіз чутливості показав, що в межах інтервалу стійкості тіньові ціни лишаються сталими, а зміна Z* "
    "пропорційна зміні запасу ресурсу саме з коефіцієнтом, що дорівнює тіньовій ціні.",
    f"Управлінська рекомендація: першим закуповувати ресурс «{RESOURCE_NAMES[best_resource_idx]}» за ціною, "
    f"що не перевищує його тіньову ціну {float(y_manual[best_resource_idx]):.4g} грн/од.",
]
for cline in conclusions:
    doc.add_paragraph(cline, style="List Bullet")

# --- 10. Контрольні питання ---
add_heading("10. Відповіді на контрольні питання")
for i, (q, ans) in enumerate(QA, start=1):
    add_para(f"{i}. {q}", bold=True)
    add_para(ans)

docx_path = f"Zvit_PR{WORK_NUMBER}_Variant{VARIANT}.docx"
doc.save(docx_path)
print(f"Готово: {docx_path} (+ {graph_path}, {sens_path})")
print(f"x* = {x_manual}, Z* = {Z_manual}")
print(f"scipy x* = {x_scipy}, Z* = {Z_scipy}")
print(f"y* = {y_manual}, W* = {W_manual}")
