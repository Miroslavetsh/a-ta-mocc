"""
Генератор звіту до практичної роботи №10 (Багатокритеріальна оптимізація: множина Парето та методи
згортки). Питає номер варіанта (або приймає його аргументом) і генерує Zvit_PR10_Variant<N>.docx.

Запуск:  .venv/bin/python generate_report.py [номер_варіанта]
"""
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
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
WORK_TITLE = "Багатокритеріальна оптимізація: множина Парето та методи згортки"
WORK_NUMBER = 10

if len(sys.argv) > 1:
    VARIANT = int(sys.argv[1])
else:
    while True:
        raw = input(f"Введи номер варіанта (1-{len(VARIANTS)}): ").strip()
        if raw.isdigit() and int(raw) in VARIANTS:
            VARIANT = int(raw)
            break
        print(f"Треба ціле число від 1 до {len(VARIANTS)}, спробуй ще раз.")


def pt(v, nd=3):
    """np.float64 -> звичайний float, округлено (інакше в numpy>=2 рядок типу 'np.float64(1.2)')."""
    if isinstance(v, (tuple, list, np.ndarray)):
        return tuple(round(float(c), nd) for c in v)
    return round(float(v), nd)


# ---------- Модель варіанта ----------
f1_c, f2_c, res, plan, weights = VARIANTS[VARIANT]
w1, w2 = weights

A_ub = [[p, q] for (p, q, _) in res] + [[-1, -1]]   # останній рядок: -(x1+x2) <= -plan
b_ub = [r for (_, _, r) in res] + [-plan]
bounds = [(0, None), (0, None)]


def f1(x):
    return float(f1_c[0] * x[0] + f1_c[1] * x[1])


def f2(x):
    return float(f2_c[0] * x[0] + f2_c[1] * x[1])


# ---------- Оптимуми за окремими критеріями: точка ідеалу і надиру ----------
r1 = linprog([-f1_c[0], -f1_c[1]], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')  # max f1
r2 = linprog([f2_c[0], f2_c[1]], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')    # min f2
if not (r1.success and r2.success):
    raise SystemExit(f"Варіант {VARIANT}: система обмежень несумісна (допустима область порожня).")

x_f1opt, x_f2opt = r1.x, r2.x
f1_ideal, f2_at_f1opt = f1(x_f1opt), f2(x_f1opt)
f1_at_f2opt, f2_ideal = f1(x_f2opt), f2(x_f2opt)
f1_nadir, f2_nadir = f1_at_f2opt, f2_at_f1opt

# ---------- Допустима область у просторі змінних (вершини многокутника) ----------
lines = [(p, q, r) for (p, q, r) in res] + [(1, 1, plan), (1, 0, 0), (0, 1, 0)]
cons = [(p, q, r, 'le') for (p, q, r) in res] + [(1, 1, plan, 'ge'), (1, 0, 0, 'ge'), (0, 1, 0, 'ge')]


def feasible(point, tol=1e-6):
    x1v, x2v = point
    for p, q, r, sense in cons:
        val = p * x1v + q * x2v
        if sense == 'le' and val > r + tol:
            return False
        if sense == 'ge' and val < r - tol:
            return False
    return True


verts = set()
for (p1, q1, rr1), (p2, q2, rr2) in itertools.combinations(lines, 2):
    A = np.array([[p1, q1], [p2, q2]])
    if abs(np.linalg.det(A)) < 1e-9:
        continue
    cand = np.linalg.solve(A, np.array([rr1, rr2]))
    if feasible(cand):
        verts.add((round(float(cand[0]), 6), round(float(cand[1]), 6)))

center = np.mean(list(verts), axis=0)
verts_sorted = sorted(verts, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
poly = np.array(verts_sorted + [verts_sorted[0]])

fig0, ax0 = plt.subplots(figsize=(6, 5))
ax0.fill(poly[:, 0], poly[:, 1], color='lightblue', alpha=.6, label='допустима область X')
ax0.plot(poly[:, 0], poly[:, 1], 'b-', lw=1)
ax0.plot([x_f2opt[0], x_f1opt[0]], [x_f2opt[1], x_f1opt[1]], 'r-', lw=3, label='множина Парето')
ax0.plot(*x_f1opt, 'g^', ms=10, label='opt f1 (max прибуток)')
ax0.plot(*x_f2opt, 'mv', ms=10, label='opt f2 (min витрати)')
ax0.set_xlabel('x1'); ax0.set_ylabel('x2'); ax0.legend(fontsize=8)
ax0.set_title(f'Варіант {VARIANT}: допустима область та множина Парето')
fig0.tight_layout()
region_path = f"feasible_region_variant_{VARIANT}.png"
fig0.savefig(region_path, dpi=150)
plt.close(fig0)

# ---------- Метод зважених сум (із нормалізацією критеріїв) ----------
g1_lo, g1_hi = -f1_ideal, -f1_nadir
g2_lo, g2_hi = f2_ideal, f2_nadir

ws_rows = []
for w in np.round(np.linspace(0, 1, 11), 2):
    c1 = w / (g1_hi - g1_lo)
    c2 = (1 - w) / (g2_hi - g2_lo)
    c = [c1 * (-f1_c[0]) + c2 * f2_c[0], c1 * (-f1_c[1]) + c2 * f2_c[1]]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    x = r.x
    ws_rows.append((float(w), x[0], x[1], f1(x), f2(x)))

ws_points = {(round(row[3], 3), round(row[4], 3)) for row in ws_rows}

# ---------- Метод ε-обмежень (не менше 6 значень) ----------
eps_list = np.round(np.linspace(f2_ideal, f2_nadir, 6), 3)
eps_rows = []
for eps in eps_list:
    A2 = A_ub + [[f2_c[0], f2_c[1]]]
    b2 = b_ub + [eps]
    r = linprog([-f1_c[0], -f1_c[1]], A_ub=A2, b_ub=b2, bounds=bounds, method='highs')
    if r.success:
        x = r.x
        eps_rows.append((float(eps), x[0], x[1], f1(x), f2(x)))

eps_points = {(round(row[3], 3), round(row[4], 3)) for row in eps_rows}

# ---------- Щільна розгортка ε як контроль "істинного" фронту + перевірка недомінованості ----------
eps_fine = np.linspace(f2_ideal, f2_nadir, 200)
front_fine = []
for eps in eps_fine:
    A2 = A_ub + [[f2_c[0], f2_c[1]]]
    b2 = b_ub + [eps]
    r = linprog([-f1_c[0], -f1_c[1]], A_ub=A2, b_ub=b2, bounds=bounds, method='highs')
    if r.success:
        front_fine.append((f1(r.x), f2(r.x)))
front_fine = np.array(front_fine)

dominated_mask = np.zeros(len(front_fine), dtype=bool)
for i in range(len(front_fine)):
    f1i, f2i = front_fine[i]
    beats = ((front_fine[:, 0] >= f1i - 1e-6) & (front_fine[:, 1] <= f2i + 1e-6) &
             ((front_fine[:, 0] > f1i + 1e-6) | (front_fine[:, 1] < f2i - 1e-6)))
    dominated_mask[i] = beats.any()
n_dominated = int(dominated_mask.sum())

fig1, ax1 = plt.subplots(figsize=(7, 5.5))
ax1.plot(front_fine[:, 0], front_fine[:, 1], 'k-', lw=2, label='фронт Парето (щільна розгортка ε)')
ax1.plot([r_[3] for r_ in ws_rows], [r_[4] for r_ in ws_rows], 'rx', ms=10, mew=2,
         label='метод зважених сум')
ax1.plot([r_[3] for r_ in eps_rows], [r_[4] for r_ in eps_rows], 'bo', ms=7, mfc='none',
         label='метод ε-обмежень')
ax1.plot(f1_ideal, f2_ideal, 'g*', ms=16, label='точка ідеалу')
ax1.plot(f1_nadir, f2_nadir, 'm*', ms=16, label='точка надиру')
ax1.set_xlabel('f1 — прибуток (max)'); ax1.set_ylabel('f2 — витрати/викиди (min)')
ax1.set_title(f'Варіант {VARIANT}: фронт Парето у просторі критеріїв')
ax1.legend(fontsize=8)
fig1.tight_layout()
front_path = f"pareto_front_variant_{VARIANT}.png"
fig1.savefig(front_path, dpi=150)
plt.close(fig1)

# ---------- Лексикографічний метод (δ = 5% за головним критерієм f1) ----------
delta = 0.05
A_lex = A_ub + [[-f1_c[0], -f1_c[1]]]
b_lex = b_ub + [-(1 - delta) * f1_ideal]
r_lex = linprog([f2_c[0], f2_c[1]], A_ub=A_lex, b_ub=b_lex, bounds=bounds, method='highs')
x_lex = r_lex.x
f1_lex, f2_lex = f1(x_lex), f2(x_lex)

# ---------- Увігнутий фронт: межа методу зважених сум (штучна задача, обидва критерії -> min) ----------
xs = np.linspace(0, 1, 201)
F_cc = np.c_[xs, 1 - xs ** 2]
cc_found = set()
for w in np.linspace(0, 1, 101):
    k = int(np.argmin(w * F_cc[:, 0] + (1 - w) * F_cc[:, 1]))
    cc_found.add(round(float(xs[k]), 3))

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(F_cc[:, 0], F_cc[:, 1], 'k-', lw=2, label='увігнутий фронт Парето (усі точки ефективні)')
found_idx = [int(np.argmin(np.abs(xs - fx))) for fx in sorted(cc_found)]
ax2.plot(F_cc[found_idx, 0], F_cc[found_idx, 1], 'rx', ms=14, mew=3, label='знайдено методом зважених сум')
ax2.set_xlabel('f1 (min)'); ax2.set_ylabel('f2 (min)')
ax2.set_title('Штучна задача з увігнутим фронтом')
ax2.legend(fontsize=8)
fig2.tight_layout()
concave_path = f"concave_front_variant_{VARIANT}.png"
fig2.savefig(concave_path, dpi=150)
plt.close(fig2)

# ---------- TOPSIS для п'яти дискретних альтернатив (точки фронту з методу ε-обмежень) ----------
alts = np.array([[r_[3], r_[4]] for r_ in eps_rows[:5]])
names = [f"A{i+1}" for i in range(len(alts))]
w_vec = np.array([w1, w2])
benefit = np.array([True, False])

N_mat = alts / np.sqrt((alts ** 2).sum(axis=0))
V_mat = N_mat * w_vec
ideal_v = np.where(benefit, V_mat.max(axis=0), V_mat.min(axis=0))
anti_v = np.where(benefit, V_mat.min(axis=0), V_mat.max(axis=0))
d_pos = np.linalg.norm(V_mat - ideal_v, axis=1)
d_neg = np.linalg.norm(V_mat - anti_v, axis=1)
C = d_neg / (d_pos + d_neg)
ranks = (-C).argsort().argsort() + 1
best_alt = names[int(np.argmax(C))]

# ---------- Чутливість рангів TOPSIS до ваг (±0.1) ----------
sens_rows = []
for dw in (-0.1, 0.0, 0.1):
    w1s = float(np.clip(w1 + dw, 0, 1)); w2s = 1 - w1s
    Vs = N_mat * np.array([w1s, w2s])
    ideal_s = np.where(benefit, Vs.max(axis=0), Vs.min(axis=0))
    anti_s = np.where(benefit, Vs.min(axis=0), Vs.max(axis=0))
    d_pos_s = np.linalg.norm(Vs - ideal_s, axis=1)
    d_neg_s = np.linalg.norm(Vs - anti_s, axis=1)
    Cs = d_neg_s / (d_pos_s + d_neg_s)
    leader = names[int(np.argmax(Cs))]
    sens_rows.append((dw, w1s, w2s, leader, [float(c) for c in Cs]))

leader_changed = len({row[3] for row in sens_rows}) > 1

# ---------- Лістинг коду (ключові фрагменти) ----------
CODE_METHODS = '''# Точка ідеалу / надиру через платіжну таблицю (дві однокритеріальні задачі ЛП)
r1 = linprog([-f1_c[0], -f1_c[1]], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')  # max f1
r2 = linprog([f2_c[0], f2_c[1]], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')    # min f2
f1_ideal, f2_nadir = f1(r1.x), f2(r1.x)
f1_nadir, f2_ideal = f1(r2.x), f2(r2.x)

# Метод зважених сум із нормалізацією критеріїв
for w in np.linspace(0, 1, 11):
    c1 = w / (g1_hi - g1_lo); c2 = (1 - w) / (g2_hi - g2_lo)      # нормалізація діапазоном ідеал/надир
    c = [c1 * (-f1_c[0]) + c2 * f2_c[0], c1 * (-f1_c[1]) + c2 * f2_c[1]]
    r = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# Метод ε-обмежень: головний критерій f1 -> max, решта критеріїв -> обмеження
for eps in np.linspace(f2_ideal, f2_nadir, 6):
    A2 = A_ub + [[f2_c[0], f2_c[1]]]; b2 = b_ub + [eps]
    r = linprog([-f1_c[0], -f1_c[1]], A_ub=A2, b_ub=b2, bounds=bounds, method='highs')

# Лексикографічний метод: головний критерій f1 фіксуємо з допуском delta, далі мінімізуємо f2
A_lex = A_ub + [[-f1_c[0], -f1_c[1]]]
b_lex = b_ub + [-(1 - delta) * f1_ideal]
r_lex = linprog([f2_c[0], f2_c[1]], A_ub=A_lex, b_ub=b_lex, bounds=bounds, method='highs')

# TOPSIS: нормалізація -> зважування -> ідеал/антиідеал -> відстані -> коефіцієнт близькості
N = alts / np.sqrt((alts ** 2).sum(axis=0))
V = N * weights
ideal = np.where(benefit, V.max(axis=0), V.min(axis=0))
anti = np.where(benefit, V.min(axis=0), V.max(axis=0))
d_pos = np.linalg.norm(V - ideal, axis=1); d_neg = np.linalg.norm(V - anti, axis=1)
C = d_neg / (d_pos + d_neg)
'''

# ---------- Контрольні питання ----------
QA = [
    ("Дайте означення домінування за Парето та Парето-оптимального рішення.",
     "Рішення x домінує за Парето рішення y (для задачі мінімізації всіх критеріїв), якщо fᵢ(x) ≤ fᵢ(y) "
     "для кожного критерію i і при цьому хоча б для одного індексу k виконується строга нерівність "
     "f_k(x) < f_k(y) — тобто x не гірше за y за жодним критерієм і строго краще хоча б за одним. "
     "Рішення називають Парето-оптимальним (ефективним), якщо серед усіх допустимих рішень не знайдеться "
     "жодного, яке б його домінувало: покращити одразу всі критерії від такої точки вже неможливо."),
    ("Чим множина Парето відрізняється від фронту Парето?",
     "Множина Парето — це сукупність самих Парето-оптимальних точок у просторі керованих змінних x "
     "(у моєму варіанті — у площині (x1, x2), це відрізок прямої між вершинами-оптимумами f1 і f2). "
     "Фронт Парето — образ цієї множини у просторі критеріїв, тобто сукупність точок (f1(x), f2(x)) для "
     "всіх x із множини Парето. Одна й та сама множина Парето може виглядати по-різному залежно від "
     "того, дивимось ми на неї в просторі рішень чи в просторі критеріїв."),
    ("Що таке точка ідеалу і точка надиру? Навіщо вони потрібні?",
     "Точка ідеалу (утопії) складається з найкращих значень кожного критерію, узятих окремо одне від "
     "одного (максимум f1 і мінімум f2 кожен сам по собі); зазвичай вона недосяжна одночасно, бо "
     "критерії суперечливі. Точка надиру складається з найгірших значень кожного критерію, але взятих "
     "саме на множині Парето (тобто найгірше значення f1 серед Парето-оптимальних рішень, і так само для "
     "f2). Обидві точки потрібні насамперед для нормалізації критеріїв різної природи й масштабу перед "
     "згорткою — без них вага фактично визначалась би одиницями виміру, а не реальними перевагами."),
    ("Чому оптимізація сама по собі не може обрати єдине рішення з фронту Парето?",
     "Усі точки фронту Парето формально рівноцінні в сенсі самого критерію Парето-оптимальності — жодна "
     "з них не домінується жодною іншою, а перехід від однієї точки до іншої завжди є компромісом: "
     "покращення одного критерію неминуче погіршує інший. Щоб з цієї множини рівноцінних (у сенсі "
     "Парето) варіантів обрати один-єдиний, потрібна додаткова інформація про переваги особи, яка "
     "приймає рішення — наприклад, ваги критеріїв, пріоритети або норма відстані до точки ідеалу, як у "
     "методі зважених сум, ε-обмежень чи TOPSIS."),
    ("Навіщо нормалізувати критерії перед застосуванням зваженої згортки?",
     "У моєму варіанті f1 вимірюється в грошових одиницях прибутку з величинами порядку тисяч, а f2 — в "
     "одиницях витрат ресурсу чи викидів порядку сотень; якщо скласти їх напряму з вагами 0,5/0,5, "
     "критерій із більшим масштабом значень фактично \"задавить\" інший незалежно від того, яку вагу він "
     "насправді має отримати. Нормалізація (наприклад, лінійне приведення до діапазону [0;1] за точками "
     "ідеалу й надиру) робить критерії порівнянними за масштабом, і тоді вага дійсно відображає задані "
     "пріоритети, а не випадкове співвідношення одиниць виміру."),
    ("Чому метод зважених сум не знаходить точок увігнутої частини фронту?",
     "Лінія рівня зваженої суми w·f1 + (1-w)·f2 = const — це пряма (у загальному випадку гіперплощина). "
     "Щоб знайти точку фронту, цю пряму \"присувають\" до допустимої області, поки вона не торкнеться її "
     "в оптимальній точці. На опуклій (угнутій знизу щодо напрямку покращення) ділянці фронту такий дотик "
     "можливий у будь-якій внутрішній точці за відповідного нахилу прямої. Але на увігнутій ділянці пряма "
     "будь-якого нахилу торкається фронту тільки по його краях — внутрішні точки дуги лежать \"глибше\", "
     "ніж пряма, що з'єднує сусідні відрізки опуклої оболонки, тому жодна вага не виділяє їх як оптимум. "
     "Це наочно підтверджує штучний приклад f1=x, f2=1-x²: 101 запуск дав лише дві точки з 201 можливої."),
    ("Опишіть метод ε-обмежень. У чому його перевага над зваженою згорткою?",
     "Один критерій (у моєму варіанті — f1, прибуток) лишають головним і оптимізують як є, а решту "
     "критеріїв переводять у розряд додаткових обмежень f2(x) ≤ ε, перебираючи ε від ідеального до "
     "надирного значення другого критерію. Кожен запуск з іншим ε дає ще одну точку фронту. Головна "
     "перевага над зваженою згорткою — метод здатен знайти будь-яку Парето-оптимальну точку, зокрема й ті, "
     "що лежать на увігнутих ділянках фронту, недосяжних для лінійної комбінації критеріїв; недолік — "
     "потрібно розв'язувати окрему задачу для кожного ε, і за надто жорсткого ε обмеження може стати "
     "несумісним із рештою системи."),
    ("Коли доцільно застосовувати лексикографічний метод?",
     "Коли критерії мають чітку, строго ієрархічну важливість для особи, яка приймає рішення: спочатку "
     "безумовно оптимізують найважливіший критерій, і лише потім, не погіршуючи (або погіршуючи не більш "
     "ніж на заданий допуск δ) досягнутий результат за ним, оптимізують наступний за важливістю критерій. "
     "У моєму варіанті це означає: прибуток f1 — головний критерій, і підприємство готове поступитися не "
     "більш ніж 5% від максимально можливого прибутку заради зменшення викидів f2. Метод підходить, коли "
     "компроміс у стилі \"трохи поступитись головним заради іншого\" неприйнятний взагалі, а не лише в "
     "принципі небажаний, як це мається на увазі у зваженій згортці."),
    ("Опишіть послідовність кроків методу TOPSIS.",
     "1) Нормалізувати матрицю рішень: rᵢⱼ = aᵢⱼ / √(Σᵢ aᵢⱼ²) — приводить стовпці різних критеріїв до "
     "порівнянного масштабу. 2) Зважити нормалізовані стовпці: vᵢⱼ = wⱼ·rᵢⱼ. 3) Визначити ідеальну "
     "альтернативу A⁺ (найкраще значення кожного критерію з урахуванням його напряму — max чи min) і "
     "антиідеальну A⁻ (найгірше значення). 4) Обчислити евклідові відстані кожної альтернативи до A⁺ і "
     "A⁻ — d⁺ і d⁻. 5) Обчислити коефіцієнт близькості Cᵢ = d⁻ᵢ/(d⁺ᵢ + d⁻ᵢ) ∈ [0;1] і впорядкувати "
     "альтернативи за його спаданням: що ближче Cᵢ до 1, то ближче альтернатива до ідеалу й далі від "
     "антиідеалу одночасно."),
    ("Чому найкраща за TOPSIS альтернатива зазвичай не є крайньою точкою фронту?",
     "Крайня точка фронту максимізує лише один критерій ціною найгіршого можливого значення іншого, тому "
     "вона одночасно і дуже близька до ідеалу за одним виміром, і дуже близька до антиідеалу за іншим — "
     "обидві відстані (d⁺ і d⁻) в сумі зазвичай не дають високого Cᵢ. Компромісна альтернатива ближче до "
     "середини фронту частіше має помірну відстань до ідеалу за обома критеріями одразу і значно більшу "
     "відстань до антиідеалу за обома критеріями, тому її коефіцієнт близькості зазвичай виявляється "
     "вищим. Це загальне правило, але не абсолютний закон: у моєму варіанті при симетричних вагах (0,5/0,5) "
     "переможцем несподівано стала саме крайня альтернатива A1 (гірша за f1, зате найкраща за f2) — такий "
     "виняток можливий, коли точки альтернатив майже лежать уздовж одного прямого відрізка (тут — єдиного "
     "ребра множини Парето) і норма відстані не завжди «штрафує» крайню точку сильніше за внутрішні; "
     "детальніше це видно з таблиці в п. 9 звіту."),
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


def fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float, np.floating)) else str(v)


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
    "Навчитися розв'язувати задачі з кількома суперечливими критеріями: будувати множину й фронт Парето, "
    "застосовувати методи зваженої суми, ε-обмежень та лексикографічного впорядкування для їх апроксимації, "
    "а також обирати остаточне рішення з дискретної множини альтернатив методом TOPSIS."
)

# --- 2. Постановка задачі ---
add_heading("2. Постановка двокритеріальної задачі, точки ідеалу та надиру")
add_para(
    f"f1(x1, x2) = {f1_c[0]}·x1 + {f1_c[1]}·x2  →  max   (прибуток)",
    bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
)
add_para(
    f"f2(x1, x2) = {f2_c[0]}·x1 + {f2_c[1]}·x2  →  min   (витрати ресурсу / викиди)",
    bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
)
cons_text = "; ".join(f"{p}x1 + {q}x2 ≤ {r}" for p, q, r in res)
add_para(f"Обмеження: {cons_text}; x1 + x2 ≥ {plan}; x1, x2 ≥ 0.")
add_para(f"Ваги критеріїв варіанта: w1 = {w1}, w2 = {w2}.")
add_para(
    f"Оптимум за f1 (max прибуток): x = {pt(x_f1opt)}, f1 = {fmt(f1_ideal)}, f2 = {fmt(f2_at_f1opt)}."
)
add_para(
    f"Оптимум за f2 (min витрати): x = {pt(x_f2opt)}, f1 = {fmt(f1_at_f2opt)}, f2 = {fmt(f2_ideal)}."
)
add_para(
    f"Точка ідеалу (f1_max; f2_min) = ({fmt(f1_ideal)}; {fmt(f2_ideal)}) — недосяжна одночасно. "
    f"Точка надиру (f1_min; f2_max) на множині Парето = ({fmt(f1_nadir)}; {fmt(f2_nadir)})."
)

# --- 3. Реалізація ---
add_heading("3. Тексти програм (ключові фрагменти)")
add_code_block(CODE_METHODS.strip("\n"))

# --- 4. Допустима область та фронт Парето ---
add_heading("4. Допустима область і фронт Парето")
doc.add_picture(region_path, width=Cm(11))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    "Допустима область у просторі змінних — чотирикутник; множина Парето (червоний відрізок) — єдине "
    "ребро цього чотирикутника між вершиною-оптимумом f1 і вершиною-оптимумом f2."
)
doc.add_picture(front_path, width=Cm(14))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    f"Образ цього ребра у просторі критеріїв і є фронтом Парето. Щільна контрольна розгортка методом "
    f"ε-обмежень (200 запусків) підтверджує: серед {len(front_fine)} побудованих точок фронту жодна не "
    f"домінується жодною іншою ({n_dominated} домінованих із {len(front_fine)}) — усі вони справді "
    f"Парето-оптимальні."
)

# --- 5. Метод зважених сум ---
add_heading("5. Метод зважених сум (із нормалізацією критеріїв)")
add_table(
    ["w", "x1", "x2", "f1", "f2"],
    [[fmt(w_, 1), fmt(x1_, 3), fmt(x2_, 3), fmt(f1_, 1), fmt(f2_, 2)] for w_, x1_, x2_, f1_, f2_ in ws_rows],
)
add_para(
    f"За 11 запусків із рівномірною сіткою ваг w = 0; 0,1; ...; 1 знайдено лише {len(ws_points)} різні "
    f"точки фронту: {sorted(ws_points)}. Причина структурна — множина Парето в цій задачі є єдиним ребром "
    f"многокутника, тому будь-яка вага, крім порогового значення, де лінія рівня паралельна цьому ребру, "
    f"виводить оптимум ЛП в один із двох його кінців."
)

# --- 6. Метод ε-обмежень ---
add_heading("6. Метод ε-обмежень і порівняння покриття фронту")
add_table(
    ["ε (f2 ≤)", "x1", "x2", "f1", "f2"],
    [[fmt(e_, 1), fmt(x1_, 3), fmt(x2_, 3), fmt(f1_, 1), fmt(f2_, 2)] for e_, x1_, x2_, f1_, f2_ in eps_rows],
)
add_para(
    f"Метод ε-обмежень за {len(eps_rows)} запусків дав {len(eps_points)} різних точок фронту, включно з "
    f"внутрішніми точками ребра, недосяжними для методу зважених сум. Це видно і на спільному графіку "
    f"вище (п. 4): точки методу зважених сум («x») групуються лише в двох кінцях відрізка, тоді як точки "
    f"методу ε-обмежень («○») рівномірно вкривають увесь фронт."
)

# --- 7. Лексикографічний метод ---
add_heading("7. Лексикографічний метод")
add_para(
    f"Головний критерій — f1 (прибуток), допуск δ = {delta:.0%}. Умова: f1(x) ≥ (1-δ)·f1_ideal = "
    f"{fmt((1 - delta) * f1_ideal)}; у межах цього допуску мінімізуємо f2."
)
add_para(
    f"Лексикографічний розв'язок: x = {pt(x_lex)}, f1 = {fmt(f1_lex)}, f2 = {fmt(f2_lex)} — порівняно з "
    f"\"жадібним\" оптимумом за самим f1 (f1 = {fmt(f1_ideal)}, f2 = {fmt(f2_at_f1opt)}) вдалось суттєво "
    f"знизити f2 ціною допустимого 5-відсоткового відхилення прибутку від максимуму."
)

# --- 8. Увігнутий фронт ---
add_heading("8. Обмеженість методу зважених сум: штучна задача з увігнутим фронтом")
doc.add_picture(concave_path, width=Cm(11))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    f"Штучна задача f1 = x → min, f2 = 1 - x² → min, x ∈ [0; 1]: усі {len(xs)} точок кривої є "
    f"Парето-оптимальними (крива строго увігнута, тобто\"випукла до початку координат\"), проте 101 "
    f"запуск методу зважених сум знайшов лише {len(cc_found)} точки: {sorted(cc_found)} — обидва кінці "
    f"відрізка. Жодна внутрішня точка дуги не стала оптимумом лінійної згортки за жодної ваги w."
)

# --- 9. TOPSIS ---
add_heading("9. Дискретний вибір альтернатив методом TOPSIS")
add_para(
    "Матриця альтернатив — п'ять точок фронту, знайдених методом ε-обмежень (п. 6); f1 — критерій "
    "вигоди (більше краще), f2 — критерій витрат (менше краще)."
)
add_table(
    ["Альт.", "f1", "f2", "d+", "d-", "C", "Ранг"],
    [[names[i], fmt(alts[i, 0], 1), fmt(alts[i, 1], 2), fmt(d_pos[i], 4), fmt(d_neg[i], 4),
      fmt(C[i], 4), int(ranks[i])] for i in range(len(names))],
)
add_para(
    f"Найкраща альтернатива за TOPSIS: {best_alt}. "
    + (
        f"Це не крайня точка фронту (нею був би {names[0]} з максимальним f1 або {names[-1]} з "
        f"мінімальним f2): компромісна альтернатива ближча до точки ідеалу у збалансованому сенсі за "
        f"обома критеріями одразу."
        if best_alt not in (names[0], names[-1]) else
        f"У цьому варіанті з обраними вагами ({w1}/{w2}) найкращою виявилась саме крайня альтернатива "
        f"{best_alt} — одна з ваг достатньо домінує, щоб переважити компромісні варіанти."
    )
)

# --- 10. Чутливість ---
add_heading("10. Чутливість рангів TOPSIS до ваг критеріїв")
add_table(
    ["Δw1", "w1", "w2", "Лідер"] + names,
    [[fmt(dw_, 1), fmt(w1s_, 2), fmt(w2s_, 2), leader_] + [fmt(c_, 4) for c_ in Cs_]
     for dw_, w1s_, w2s_, leader_, Cs_ in sens_rows],
)
add_para(
    ("Лідер ранжування змінився при зміщенні ваг на ±0,1: це означає, що вибір найкращої альтернативи в "
     "цьому варіанті чутливий до того, наскільки точно ОПР визначив співвідношення важливості критеріїв, "
     "і обґрунтуванню ваг варто приділити окрему увагу.")
    if leader_changed else
    ("Лідер ранжування не змінився при зміщенні ваг на ±0,1: обрана альтернатива стійка до помірних "
     "похибок у визначенні ваг критеріїв, тому рішення можна вважати надійним навіть за не цілком точно "
     "заданих пріоритетів.")
)

# --- 11. Висновки ---
add_heading("11. Висновки")
conclusions = [
    f"Для варіанта {VARIANT} точка ідеалу ({fmt(f1_ideal)}; {fmt(f2_ideal)}) недосяжна — максимальний "
    f"прибуток і мінімальні витрати ресурсу не досягаються одночасно, що кількісно підтверджує конфлікт "
    f"критеріїв f1 і f2.",
    f"Метод зважених сум за 11 запусків знайшов лише {len(ws_points)} різні точки фронту, оскільки "
    f"множина Парето цієї задачі ЛП — єдине ребро допустимого многокутника, а оптимум лінійної задачі "
    f"завжди припадає на вершину.",
    f"Метод ε-обмежень за {len(eps_rows)} запусків дав {len(eps_points)} різних точок, включно з "
    f"внутрішніми точками ребра, і здатен апроксимувати фронт довільної форми, зокрема увігнутої — "
    f"це наочно показала штучна задача з п. 8, де зважена сума знайшла лише кінці відрізка.",
    f"Лексикографічний метод із допуском δ = 5% дав компромісну точку f1 = {fmt(f1_lex)}, "
    f"f2 = {fmt(f2_lex)}, відмінну і від чистого оптимуму за f1, і від оптимуму за f2 — корисний спосіб "
    f"формалізувати сувору ієрархію критеріїв без повного перебору ваг.",
    f"TOPSIS із вагами ({w1}/{w2}) обрав альтернативу {best_alt} із п'яти точок фронту; аналіз чутливості "
    + ("показав, що цей вибір нестійкий до відхилення ваг на ±0,1." if leader_changed else
       "показав, що цей вибір стійкий до відхилення ваг на ±0,1."),
    "Загальна рекомендація: якщо форма фронту й опуклість заздалегідь невідомі (а в задачах складних "
    "систем це типово), варто спиратись на метод ε-обмежень або чебишовську згортку, а не лише на зважену "
    "суму; лексикографічний метод доречний за строгої ієрархії критеріїв, а TOPSIS — коли остаточний вибір "
    "потрібно зробити з уже готової дискретної множини варіантів.",
]
for c in conclusions:
    doc.add_paragraph(c, style="List Bullet")

# --- 12. Контрольні питання ---
add_heading("12. Відповіді на контрольні питання")
for i, (q, ans) in enumerate(QA, start=1):
    add_para(f"{i}. {q}", bold=True)
    add_para(ans)

docx_path = f"Zvit_PR{WORK_NUMBER}_Variant{VARIANT}.docx"
doc.save(docx_path)
print(f"Готово: {docx_path} (+ {region_path}, {front_path}, {concave_path})")
