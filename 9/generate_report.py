"""
Генератор звіту до практичної роботи №9 (Динамічне програмування: принцип оптимальності Беллмана).
Питає номер варіанта (або приймає його аргументом) і генерує Zvit_PR9_Variant<N>.docx
з усіма розрахунками, графіками і відповідями на контрольні питання саме під цей варіант.

Запуск:  .venv/bin/python generate_report.py [номер_варіанта]
"""
import sys
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
WORK_TITLE = "Динамічне програмування: принцип оптимальності Беллмана"
WORK_NUMBER = 9

if len(sys.argv) > 1:
    VARIANT = int(sys.argv[1])
else:
    while True:
        raw = input(f"Введи номер варіанта (1-{len(VARIANTS)}): ").strip()
        if raw.isdigit() and int(raw) in VARIANTS:
            VARIANT = int(raw)
            break
        print(f"Треба ціле число від 1 до {len(VARIANTS)}, спробуй ще раз.")

# ---------- Дані варіанта ----------
g_raw, R, STAGES = VARIANTS[VARIANT]
g = np.array(g_raw, float)
N = g.shape[0]

# ---------- Задача 1. Розподіл ресурсу: зворотний хід Беллмана ----------
F = np.zeros((N + 1, R + 1))
U = np.zeros((N + 1, R + 1), int)
for k in range(1, N + 1):
    for s in range(R + 1):
        vals = [g[k - 1, u] + F[k - 1, s - u] for u in range(s + 1)]
        U[k, s] = int(np.argmax(vals))
        F[k, s] = max(vals)

# ---------- Прямий хід: відновлення оптимального плану ----------
plan = []
s = R
for k in range(N, 0, -1):
    u_k = int(U[k, s])
    plan.append(u_k)
    s -= u_k
plan.reverse()
total_profit = float(F[N, R])

# ---------- Повний перебір (перевірка) ----------
best_bf = -1.0
best_plan_bf = None
for combo in itertools.product(range(R + 1), repeat=N):
    if sum(combo) != R:
        continue
    val = float(sum(g[i, combo[i]] for i in range(N)))
    if val > best_bf:
        best_bf, best_plan_bf = val, combo
assert abs(best_bf - total_profit) < 1e-9, "ДП і повний перебір розійшлись!"

# ---------- Порівняння з рівномірним і жадібним розподілами ----------
base = R // N
rem = R - base * N
uniform_profit = float(sum(g[i, base] for i in range(N)))
greedy_vals = [float(g[i, R]) for i in range(N)]
greedy_best = int(np.argmax(greedy_vals))
greedy_profit = greedy_vals[greedy_best]
gain_uniform = (total_profit - uniform_profit) / uniform_profit * 100
gain_greedy = (total_profit - greedy_profit) / greedy_profit * 100

# ---------- Задача 2. Власна багатоетапна мережа: пошук найкоротшого шляху ----------
def layer_sizes(stages):
    """Шари мережі так, щоб вершин було не менше 9 (як у прикладі з методички)."""
    presets = {3: [1, 3, 4, 1], 4: [1, 2, 3, 2, 1], 5: [1, 2, 2, 2, 1, 1]}
    return presets[stages]


def build_network(stages, variant):
    sizes = layer_sizes(stages)
    letters = "BCDEFGHIJKLMNOPQRSTUVWXY"
    layers = []
    for li, sz in enumerate(sizes):
        if li == 0:
            layers.append(["A"])
        elif li == len(sizes) - 1:
            layers.append(["Z"])
        else:
            layers.append([f"{letters[li - 1]}{j + 1}" for j in range(sz)])
    edges = {v: {} for layer in layers for v in layer}
    for li in range(len(layers) - 1):
        for i, u in enumerate(layers[li]):
            for j, v in enumerate(layers[li + 1]):
                w = ((i + 1) * 3 + (j + 1) * 5 + variant * 2 + li * 7) % 12 + 1
                edges[u][v] = w
    return layers, edges


layers, edges = build_network(STAGES, VARIANT)
n_vertices = sum(len(l) for l in layers)
all_nodes = [v for layer in layers for v in layer]
start, end = layers[0][0], layers[-1][0]
rev_order = list(reversed(all_nodes))

Vb = {end: 0.0}
nxt = {}
for v in rev_order[1:]:
    Vb[v], nxt[v] = min(((edges[v][w] + Vb[w], w) for w in edges[v]), key=lambda t: t[0])

path = [start]
v = start
while v != end:
    v = nxt[v]
    path.append(v)
shortest_len = Vb[start]


def all_paths(u, target):
    if u == target:
        yield [u]
        return
    for w in edges[u]:
        for tail in all_paths(w, target):
            yield [u] + tail


best_len_bf = min(sum(edges[p[i]][p[i + 1]] for i in range(len(p) - 1)) for p in all_paths(start, end))
assert best_len_bf == shortest_len, "ДП і повний перебір шляхів розійшлись!"

# жадібний шлях: на кожному кроці йти ребром із мінімальною вагою, без огляду на V(w)
greedy_path = [start]
vg = start
while vg != end:
    vg = min(edges[vg], key=edges[vg].get)
    greedy_path.append(vg)
greedy_len = sum(edges[greedy_path[i]][greedy_path[i + 1]] for i in range(len(greedy_path) - 1))
greedy_matches = (greedy_path == path)
if greedy_matches:
    greedy_note = (
        f"У цій конкретній мережі жадібний вибір найдешевшого ребра на кожному кроці випадково збігся з "
        f"оптимальним шляхом: {' -> '.join(greedy_path)}, довжина {greedy_len:.0f} = {shortest_len:.0f}. Це не "
        f"закономірність, а збіг для цих конкретних ваг: рішення на кожному кроці все одно приймається за сумою "
        f"c(v,w) + V(w), а не за самим c(v,w) — просто в цій мережі порядок ребер за вагою з вершини A, B2 і C2 "
        f"випадково узгоджується з порядком за V(w). Якби ваги були іншими, найдешевше ребро могло б вести у "
        f"вершину з великим V(w) і програвати трохи дорожчій альтернативі з набагато меншим V(w) — так і сталося "
        f"б у загальному випадку (приклад — мережа з методички: там теж немає гарантії, що локально дешевше "
        f"ребро виграє в сумі)."
    )
else:
    greedy_note = (
        f"Жадібний вибір найдешевшого ребра на кожному кроці дає інший, довший шлях: "
        f"{' -> '.join(greedy_path)}, довжина {greedy_len:.0f} > {shortest_len:.0f} — гірше за оптимум ДП. Це "
        f"наочно показує, що рішення на кожному кроці потрібно приймати за сумою c(v,w) + V(w) (вартість кроку + "
        f"довжина оптимального залишку шляху), а не лише за вартістю самого кроку c(v,w)."
    )

# схема мережі з виділеним найкоротшим шляхом
fig_net, ax_net = plt.subplots(figsize=(10, 5))
pos = {}
for li, layer in enumerate(layers):
    for j, vv in enumerate(layer):
        y = j - (len(layer) - 1) / 2
        pos[vv] = (li, y)
for u_ in edges:
    for vv, w in edges[u_].items():
        x1_, y1_ = pos[u_]; x2_, y2_ = pos[vv]
        on_path = u_ in path and vv in path and path.index(vv) == path.index(u_) + 1
        ax_net.plot([x1_, x2_], [y1_, y2_], color='red' if on_path else 'gray',
                    lw=2.5 if on_path else 1, zorder=1 if on_path else 0)
        ax_net.annotate(str(w), ((x1_ + x2_) / 2, (y1_ + y2_) / 2), fontsize=8, color='black')
for vv, (x, y) in pos.items():
    ax_net.scatter([x], [y], s=500, color='lightgreen' if vv in path else 'lightblue', zorder=2, edgecolors='black')
    ax_net.annotate(vv, (x, y), ha='center', va='center', fontsize=9, zorder=3)
ax_net.set_title(f"Варіант {VARIANT}: мережа ({STAGES} етапи), найкоротший шлях {' -> '.join(path)}")
ax_net.axis('off')
fig_net.tight_layout()
network_path = f"network_variant_{VARIANT}.png"
fig_net.savefig(network_path, dpi=150)
plt.close(fig_net)

# ---------- Задача 3 (додаткова). Заміна обладнання за принципом Беллмана ----------
p_age = [100, 90, 75, 55, 30]
c_age = [10, 15, 25, 40, 60]
REPLACE_COST = 50
HORIZON = 5
MAX_AGE = 4


def revenue_keep(a):
    return p_age[a] - c_age[a]


revenue_replace = p_age[0] - c_age[0] - REPLACE_COST

Veq = {0: {a: 0.0 for a in range(MAX_AGE + 1)}}
dec = {}
for k in range(1, HORIZON + 1):
    Veq[k] = {}
    dec[k] = {}
    for a in range(MAX_AGE + 1):
        options = []
        if a < MAX_AGE:
            options.append(("зберегти", revenue_keep(a) + Veq[k - 1][a + 1]))
        options.append(("замінити", revenue_replace + Veq[k - 1][1]))
        dec[k][a], Veq[k][a] = max(options, key=lambda t: t[1])

a_ = 0
plan_eq = []
total_eq = 0.0
for k in range(HORIZON, 0, -1):
    d = dec[k][a_]
    gain = revenue_keep(a_) if d == "зберегти" else revenue_replace
    plan_eq.append((HORIZON - k + 1, a_, d, gain))
    total_eq += gain
    a_ = a_ + 1 if d == "зберегти" else 1

best_eq = -1.0
best_seq = None
for seq in itertools.product([0, 1], repeat=HORIZON):
    age = 0
    total_b = 0.0
    ok = True
    for d in seq:
        if age == MAX_AGE and d == 0:
            ok = False
            break
        if d == 0:
            total_b += revenue_keep(age)
            age += 1
        else:
            total_b += revenue_replace
            age = 1
    if ok and total_b > best_eq:
        best_eq, best_seq = total_b, seq
assert abs(best_eq - total_eq) < 1e-9, "ДП і повний перебір заміни обладнання розійшлись!"

# ---------- Складність ДП проти повного перебору + графік ----------
Rs = list(range(5, 31, 5))
dp_ops = [N * (Rv + 1) * (Rv + 2) // 2 for Rv in Rs]
brute_ops = [int(math.comb(Rv + N - 1, N - 1)) for Rv in Rs]

fig_c, ax_c = plt.subplots(figsize=(7, 5))
ax_c.plot(Rs, dp_ops, 'o-', label='ДП: N*(R+1)*(R+2)/2')
ax_c.plot(Rs, brute_ops, 's-', label='Повний перебір: C(R+N-1, N-1)')
ax_c.set_yscale('log')
ax_c.set_xlabel('R (обсяг ресурсу)')
ax_c.set_ylabel('Кількість обчислень (лог. шкала)')
ax_c.set_title(f'Варіант {VARIANT}: зростання трудомісткості при N={N}')
ax_c.legend()
fig_c.tight_layout()
complexity_path = f"complexity_variant_{VARIANT}.png"
fig_c.savefig(complexity_path, dpi=150)
plt.close(fig_c)

# ---------- Додаткове завдання (на "відмінно"): R=10, лінійно продовжені таблиці ----------
def extend_linear(row, new_len):
    row = list(row)
    last_diff = row[-1] - row[-2]
    while len(row) < new_len:
        row.append(row[-1] + last_diff)
    return row


R_bonus = 10
g_bonus = np.array([extend_linear(g[i], R_bonus + 1) for i in range(N)], float)

F_b = np.zeros((N + 1, R_bonus + 1))
U_b = np.zeros((N + 1, R_bonus + 1), int)
for k in range(1, N + 1):
    for s in range(R_bonus + 1):
        vals = [g_bonus[k - 1, u] + F_b[k - 1, s - u] for u in range(s + 1)]
        U_b[k, s] = int(np.argmax(vals))
        F_b[k, s] = max(vals)

plan_b, s_b = [], R_bonus
for k in range(N, 0, -1):
    u_k = int(U_b[k, s_b])
    plan_b.append(u_k)
    s_b -= u_k
plan_b.reverse()
total_bonus = float(F_b[N, R_bonus])

best_bf_b = -1.0
best_plan_bf_b = None
for combo in itertools.product(range(R_bonus + 1), repeat=N):
    if sum(combo) != R_bonus:
        continue
    val = float(sum(g_bonus[i, combo[i]] for i in range(N)))
    if val > best_bf_b:
        best_bf_b, best_plan_bf_b = val, combo
assert abs(best_bf_b - total_bonus) < 1e-9, "ДП і повний перебір для R=10 розійшлись!"

# ---------- Хелпери форматування (уникаємо np.float64(...) у тексті) ----------
def ff(v, nd=0):
    return f"{float(v):.{nd}f}"


def tup(seq, nd=0):
    return "(" + ", ".join(ff(x, nd) for x in seq) + ")"


# ---------- Лістинг коду ----------
CODE_METHODS = '''# F[k][s] -- умовний оптимальний виграш перших k підприємств при ресурсі s
# U[k][s] -- умовне оптимальне керування (скільки виділити k-му підприємству)
F = np.zeros((N + 1, R + 1))
U = np.zeros((N + 1, R + 1), int)
for k in range(1, N + 1):                      # зворотний хід Беллмана
    for s in range(R + 1):
        vals = [g[k-1][u] + F[k-1][s-u] for u in range(s + 1)]
        U[k][s] = int(np.argmax(vals))
        F[k][s] = max(vals)

plan, s = [], R                                # прямий хід: відновлення плану
for k in range(N, 0, -1):
    plan.append(int(U[k][s])); s -= U[k][s]
plan.reverse()

# Найкоротший шлях у багатоетапній мережі (функції Беллмана V(v))
V, nxt = {end: 0.0}, {}
for v in reversed_topological_order:           # від кінцевої вершини до початкової
    V[v], nxt[v] = min((c + V[w], w) for w, c in edges[v].items())
path, v = [start], start
while v != end:
    v = nxt[v]; path.append(v)
'''

RESULT_OUTPUT = (
    f"Таблиця умовних оптимальних виграшів F_k(s):\n"
    + "  s " + "".join(f"{s:>7}" for s in range(R + 1)) + "\n"
    + "\n".join(
        "F_" + str(k) + " " + "".join(f"{F[k][s]:>7.0f}" for s in range(R + 1))
        for k in range(1, N + 1)
    )
    + f"\n\nоптимальний план розподілу: {list(plan)}, сумарний прибуток = {total_profit:.0f}\n"
    + f"рівномірний розподіл дав би: {uniform_profit:.0f} (по {base} од. кожному, {rem} од. не використана)\n"
    + f"жадібний розподіл дав би: {greedy_profit:.0f} (усе підприємству {greedy_best + 1})\n"
    + f"кількість обчислень: ДП {N * (R + 1) * (R + 2) // 2} проти повного перебору "
    + f"{int(math.comb(R + N - 1, N - 1))}\n\n"
    + f"функції Беллмана мережі: " + ", ".join(f"V({v})={Vb[v]:.0f}" for v in reversed(all_nodes)) + "\n"
    + f"найкоротший шлях: {' -> '.join(path)}, довжина = {shortest_len:.0f}"
)

# ---------- Контрольні питання ----------
QA = [
    ("Сформулюйте принцип оптимальності Беллмана.",
     "Яким би не був початковий стан і початкове рішення на першому кроці, усі наступні рішення мають утворювати "
     "оптимальну стратегію відносно того стану, що виник після цього першого рішення. Інакше кажучи, будь-яка "
     "«хвостова» частина оптимальної траєкторії сама є оптимальною траєкторією для підзадачі, що залишилась. Саме "
     "ця властивість дозволяє не перебирати всі стратегії цілком, а розв'язувати задачу послідовно, крок за кроком, "
     "зберігаючи лише умовно оптимальні результати для кожного стану."),
    ("Що таке стан, керування і функція Беллмана в багатокроковій задачі?",
     "Стан sₖ — це набір чисел, що повністю описує систему перед k-м кроком і є достатнім для прийняття подальших "
     "рішень (у задачі розподілу ресурсу — залишок нерозподіленого ресурсу; у задачі найкоротшого шляху — поточна "
     "вершина мережі). Керування uₖ — рішення, яке приймається на k-му кроці (скільки ресурсу виділити, яку "
     "вершину обрати наступною). Функція Беллмана Fₖ(s) (або V(v)) — це накопичений оптимальний результат "
     "(максимальний виграш чи мінімальна довжина шляху), який можна отримати з даного стану за умови, що всі "
     "подальші рішення обираються оптимально."),
    ("Які дві властивості задачі необхідні для застосування динамічного програмування?",
     "По-перше, задача повинна мати оптимальну підструктуру: оптимальний розв'язок усієї задачі можна виразити "
     "через оптимальні розв'язки її підзадач (саме це гарантує принцип оптимальності Беллмана). По-друге, підзадачі "
     "мають перекриватися: одні й ті самі підзадачі (пари «крок, стан») виникають багаторазово в різних "
     "траєкторіях, тому їх варто обчислити й запам'ятати один раз, а не перераховувати щоразу — якщо б підзадачі "
     "не повторювались, вигоди від динамічного програмування порівняно зі звичайною рекурсією не було б."),
    ("Чим відрізняється зворотний хід від прямого і навіщо потрібні обидва?",
     "Зворотний хід рухається від останнього кроку до першого (чи від кінцевої вершини до початкової) і для "
     "кожного можливого стану обчислює умовно оптимальний виграш Fₖ(s) та відповідне умовне керування uₖ*(s) — "
     "результат залежить від стану, а не від того, як система в нього потрапила. Але сам зворотний хід не дає "
     "єдиного числового плану — він дає лише таблицю «що робити в стані s на кроці k». Прямий хід рухається від "
     "першого кроку до останнього (чи навпаки, залежно від напрямку нумерації), підставляючи у ці таблиці фактичний "
     "початковий стан (весь ресурс R, стартову вершину) і послідовно «прочитуючи» вже знайдені умовно оптимальні "
     "керування, щоб відновити один конкретний оптимальний план дій."),
    ("Запишіть рівняння Беллмана для задачі розподілу ресурсу.",
     "Fₖ(s) = max за u від 0 до s { gₖ(u) + Fₖ₋₁(s − u) }, з початковою умовою F₀(s) = 0 для будь-якого s. Тут "
     "Fₖ(s) — максимальний сумарний прибуток перших k підприємств при ресурсі s, gₖ(u) — табличний прибуток k-го "
     "підприємства від вкладення u одиниць, а s − u — залишок ресурсу, що передається на подальші (k−1) "
     "підприємств."),
    ("Чому оптимальний розподіл ресурсу не збігається з рівномірним?",
     f"Бо прибуток кожного підприємства росте з вкладенням нелінійно (неоднаковими приростами), тож однакова "
     f"одиниця ресурсу дає різний додатковий прибуток у різних підприємств і на різних рівнях вкладення. Для "
     f"варіанта {VARIANT} рівномірний розподіл по {base} од. кожному дає лише {uniform_profit:.0f}, тоді як "
     f"оптимальний план {tup(plan)} — {total_profit:.0f}, тобто на {gain_uniform:.1f}% більше: ресурс варто "
     f"спрямовувати туди, де гранична (додаткова) віддача від останньої одиниці найвища, а не розподіляти його "
     f"механічно рівними частками."),
    ("Чому жадібний вибір найдешевшого ребра не завжди дає найкоротший шлях?",
     f"Тому що найдешевше ребро з поточної вершини веде в сусідню вершину, з якої в загальному випадку може "
     f"залишитись довгий і дорогий залишок шляху до кінця, тоді як трохи дорожче ребро може вести у вершину з "
     f"набагато коротшим залишком. Рішення потрібно приймати не за вартістю одного кроку c(v,w), а за сумою "
     f"c(v,w) + V(w) — вартості кроку й функції Беллмана (довжини оптимального залишку шляху) наступної вершини; "
     f"саме так і реалізує принцип Беллмана відмову від «жадібної» евристики в загальному випадку. {greedy_note}"),
    ("Порівняйте складність ДП і повного перебору для задачі розподілу R одиниць між N підрозділами.",
     f"Повний перебір розглядає всі композиції числа R на N невід'ємних доданків — це C(R+N−1, N−1) варіантів, що "
     f"росте експоненційно (для N={N}, R={R} це {int(math.comb(R + N - 1, N - 1))} варіантів, а для R=20, N=8 — "
     f"уже близько 21⁸ ≈ 3,8·10¹⁰). Динамічне програмування обчислює таблицю з (N+1)×(R+1) клітинок, і для кожної "
     f"клітинки перебір u коштує O(R), тож загальна складність — O(N·R²) (орієнтовно {N * (R + 1) * (R + 2) // 2} "
     f"операцій у нашому випадку). Різниця — заміна експоненційного росту за N на поліноміальний, досягнута "
     f"повторним використанням розв'язків підзадач замість перебору всіх комбінацій заново."),
    ("Що таке прокляття розмірності? Як воно обмежує застосування ДП?",
     "Прокляттям розмірності Беллман назвав те, що складність динамічного програмування зростає як O(Rᵃ), де d — "
     "розмірність вектора стану. Якщо стан — одне число (залишок ресурсу, вік машини, номер вершини), складність "
     "лишається поліноміальною й метод працює чудово. Але щойно стан стає вектором із кількох незалежних "
     "компонент (наприклад, залишки кількох видів ресурсу одночасно), кількість станів росте як добуток їхніх "
     "діапазонів, тобто експоненційно за кількістю компонент — і вже при d = 4-5 пряме табулювання стає "
     "обчислювально неможливим навіть при помірних R. Це обмежує пряме застосування класичного табличного ДП "
     "задачами з невеликою (1-3) розмірністю стану; для більших розмірностей використовують наближені методи "
     "(наприклад, апроксимацію функції Беллмана)."),
    ("Наведіть приклади задач управління складними системами, які зводяться до схеми ДП.",
     "Розподіл обмеженого інвестиційного чи виробничого ресурсу між підрозділами (як у цій роботі); пошук "
     "найкоротшого чи найдешевшого маршруту в транспортній або телекомунікаційній мережі (а також споріднені "
     "алгоритми Дейкстри й Беллмана-Форда); політика заміни обладнання чи транспорту залежно від його віку; "
     "управління запасами на складі (скільки замовляти щопериоду, щоб мінімізувати сумарні витрати на зберігання "
     "й дефіцит); задача про рюкзак і споріднені задачі календарного планування виробництва по періодах; "
     "розподіл бюджету рекламної кампанії між каналами по тижнях. Усі ці задачі природно розпадаються на "
     "послідовність кроків із чітким станом, набором керувань і перекривними підзадачами."),
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


def fmt(v, nd=0):
    return f"{float(v):.{nd}f}" if isinstance(v, (int, float, np.floating)) else str(v)


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
    "Опанувати метод динамічного програмування для багатокрокових задач прийняття рішень: навчитися будувати "
    "рекурентні співвідношення Беллмана, виконувати зворотний і прямий хід та застосовувати метод до задач "
    "розподілу ресурсу, пошуку найкоротшого шляху та заміни обладнання; порівняти обчислювальну складність "
    "динамічного програмування з повним перебором."
)

# --- 2. Формалізація задачі ---
add_heading("2. Формалізація задачі розподілу ресурсу")
add_para(
    f"Варіант {VARIANT}: R = {int(R)} одиниць ресурсу розподіляються між N = {N} підприємствами; прибуток "
    f"кожного підприємства від вкладення u = 0..5 одиниць заданий таблично:"
)
add_table(
    ["Підприємство \\ u"] + [str(u) for u in range(g.shape[1])],
    [[f"№{i + 1}"] + [fmt(g[i, u]) for u in range(g.shape[1])] for i in range(N)]
)
add_para(
    "Крок k = 1, 2, 3, 4 — черговість, у якій розглядається k-те підприємство (порядок умовний і на результат не "
    "впливає). Стан sₖ — залишок нерозподіленого ресурсу перед кроком k, sₖ ∈ {0, 1, ..., R}. Керування uₖ — "
    "обсяг ресурсу, що виділяється k-му підприємству на цьому кроці, 0 ≤ uₖ ≤ sₖ. Функція переходу: "
    "sₖ₊₁ = sₖ − uₖ. Виграш кроку: gₖ(uₖ) — табличний прибуток k-го підприємства від вкладення uₖ одиниць."
)

# --- 3. Рекурентне співвідношення ---
add_heading("3. Рекурентне співвідношення Беллмана")
add_para(
    "Fₖ(s) = max за u від 0 до s { gₖ(u) + Fₖ₋₁(s − u) },   F₀(s) = 0.",
    bold=True, align=WD_ALIGN_PARAGRAPH.CENTER
)
add_para(
    "Fₖ(s) — максимальний сумарний прибуток перших k підприємств, якщо в розпорядженні є s одиниць ресурсу. Разом "
    "зі значенням Fₖ(s) на кожному кроці зберігають умовне оптимальне керування uₖ*(s) — те u, на якому "
    "досягається максимум; саме ці таблиці й заповнює зворотний хід нижче."
)

# --- 4. Повні таблиці Fk(s), Uk(s) ---
add_heading("4. Таблиці умовних оптимальних виграшів Fₖ(s) і керувань uₖ*(s)")
add_table(
    ["k \\ s"] + [f"s={s}" for s in range(R + 1)],
    [[f"F_{k}"] + [fmt(F[k, s]) for s in range(R + 1)] for k in range(1, N + 1)]
)
add_table(
    ["k \\ s"] + [f"s={s}" for s in range(R + 1)],
    [[f"u_{k}*"] + [fmt(U[k, s]) for s in range(R + 1)] for k in range(1, N + 1)]
)
k_demo, s_demo = 2, R
vals_demo = [(u, g[k_demo - 1, u], F[k_demo - 1, s_demo - u]) for u in range(s_demo + 1)]
demo_lines = "\n".join(
    f"u={u}: g_{k_demo}({u})={gu:.0f} + F_{k_demo - 1}({s_demo - u})={fprev:.0f}  ->  {gu + fprev:.0f}"
    for u, gu, fprev in vals_demo
)
add_para(f"Приклад обчислення клітинки вручну — F_{k_demo}({s_demo}):", bold=True)
add_code_block(demo_lines)
add_para(
    f"Максимум серед цих варіантів дає F_{k_demo}({s_demo}) = {F[k_demo, s_demo]:.0f} при u*_{k_demo}({s_demo}) = "
    f"{U[k_demo, s_demo]} — саме це значення і записане в таблиці вище."
)

# --- 5. Прямий хід ---
add_heading("5. Прямий хід і відновлений оптимальний план")
add_para(
    "Прямий хід рухається від останнього кроку до першого: беремо повний ресурс s = R, зчитуємо u_N*(R), "
    "зменшуємо залишок на цю величину, переходимо до кроку N−1, і так далі до кроку 1."
)
add_para(
    f"Оптимальний план розподілу: (u1, u2, u3, u4) = {tup(plan)}. Сумарний прибуток: "
    f"F_{N}({R}) = {total_profit:.0f}.",
    bold=True
)

# --- 6. Текст програми та результати ---
add_heading("6. Текст програми та результати її роботи")
add_code_block(CODE_METHODS.strip("\n"))
add_para("Результат роботи програми:", bold=True)
add_code_block(RESULT_OUTPUT)
add_para(
    f"Повний перебір усіх {int(math.comb(R + N - 1, N - 1))} допустимих планів (combo з sum(combo)=R) підтвердив "
    f"той самий оптимум {best_bf:.0f} при плані {tup(best_plan_bf)}, що й динамічне програмування — отже реалізація "
    f"коректна."
)

# --- 7. Порівняння з рівномірним і жадібним розподілами ---
add_heading("7. Порівняння оптимального плану з рівномірним і жадібним")
add_table(
    ["Стратегія", "План", "Прибуток", "Виграш ДП, %"],
    [
        ["Динамічне програмування", tup(plan), fmt(total_profit), "—"],
        ["Рівномірний", f"по {base} кожному ({rem} не використано)", fmt(uniform_profit), f"{gain_uniform:.2f}"],
        ["Жадібний", f"усе підприємству {greedy_best + 1}", fmt(greedy_profit), f"{gain_greedy:.2f}"],
    ]
)
add_para(
    f"Оптимальний план дає на {gain_uniform:.1f}% більший прибуток, ніж рівномірний розподіл, і на "
    f"{gain_greedy:.1f}% більший, ніж жадібна стратегія «усе одному найприбутковішому підприємству». Це "
    f"ілюструє головну ідею ДП: ресурс потрібно розподіляти за граничною (маржинальною) віддачею кожної наступної "
    f"одиниці для кожного підприємства, а не механічно рівними частками чи повністю одному напрямку."
)

# --- 8. Власна мережа ---
add_heading("8. Власна багатоетапна мережа і пошук найкоротшого шляху")
add_para(
    f"Побудована мережа з {STAGES} етапів і {n_vertices} вершин (не менше 9, як вимагає завдання). Вага ребра — "
    f"довжина відповідної ділянки шляху."
)
doc.add_picture(network_path, width=Cm(16))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    "Функції Беллмана V(v) — довжина найкоротшого шляху від вершини v до кінцевої вершини Z — обчислені у "
    "зворотному топологічному порядку за формулою V(Z)=0, V(v) = min по (v,w) з E { c(v,w) + V(w) }:"
)
add_table(
    [""] + list(reversed(all_nodes)),
    [["V(·)"] + [fmt(Vb[v]) for v in reversed(all_nodes)]]
)
add_para(
    f"Найкоротший шлях: {' -> '.join(path)}, довжина = {shortest_len:.0f}. Повний перебір усіх шляхів з {start} до "
    f"{end} підтвердив той самий результат ({best_len_bf:.0f}).",
    bold=True
)
add_para(greedy_note)

# --- 9. Додаткова задача: заміна обладнання ---
add_heading("9. Додаткова задача: заміна обладнання за принципом Беллмана")
add_para(
    "Модель (єдина ілюстративна модель, спільна для всіх варіантів): вік машини a = 0..4 роки, прибуток за рік "
    f"p(a) = {[int(x) for x in p_age]}, витрати на обслуговування c(a) = {[int(x) for x in c_age]}, вартість "
    f"заміни = {REPLACE_COST}, горизонт планування = {HORIZON} років. На початку кожного року обирається рішення: "
    "«зберегти» (вік зростає на 1, дохід = p(a) − c(a)) або «замінити» (машина одразу стає новою, дохід = "
    "p(0) − c(0) − вартість заміни, вік після заміни = 1); у віці 4 заміна обов'язкова."
)
add_table(
    ["Рік", "Вік на початку року", "Рішення", "Прибуток року"],
    [[str(year), str(age), d, fmt(gain)] for year, age, d, gain in plan_eq]
)
add_para(
    f"Сумарний прибуток за {HORIZON} років за цим планом: {total_eq:.0f}. Повний перебір усіх "
    f"{2 ** HORIZON} послідовностей «зберегти/замінити» підтвердив той самий оптимум ({best_eq:.0f}).",
    bold=True
)

# --- 10. Оцінка складності ---
add_heading("10. Оцінка обчислювальної складності ДП і повного перебору")
add_table(
    ["R", "ДП, операцій O(N·R²)", "Повний перебір, варіантів"],
    [[str(Rv), str(d), str(b)] for Rv, d, b in zip(Rs, dp_ops, brute_ops)]
)
doc.add_picture(complexity_path, width=Cm(13))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    f"Для варіанта {VARIANT} (N={N}) при зростанні R з {Rs[0]} до {Rs[-1]} кількість операцій ДП зростає лише в "
    f"{dp_ops[-1] / dp_ops[0]:.1f} раза, тоді як кількість варіантів повного перебору — у "
    f"{brute_ops[-1] / brute_ops[0]:.0f} разів; на логарифмічній шкалі графіка це видно як лінійне зростання кривої "
    f"ДП проти значно крутішого зростання кривої повного перебору."
)
add_para(
    f"Додаткове завдання (на «відмінно»): таблиці прибутків лінійно продовжені до u = 10 (останнім відомим "
    f"граничним приростом), і задача розв'язана для R = {R_bonus}. Оптимальний план: {tup(plan_b)}, прибуток "
    f"{total_bonus:.0f} — підтверджено повним перебором із {int(math.comb(R_bonus + N - 1, N - 1))} варіантів."
)

# --- 11. Висновки ---
add_heading("11. Висновки")
conclusions = [
    f"Для варіанта {VARIANT} динамічне програмування знайшло оптимальний план розподілу ресурсу {tup(plan)} із "
    f"прибутком {total_profit:.0f}, що підтверджено повним перебором і перевищує рівномірний розподіл на "
    f"{gain_uniform:.1f}%, а жадібний — на {gain_greedy:.1f}%.",
    f"У власній багатоетапній мережі з {STAGES} етапів і {n_vertices} вершинами знайдений методом Беллмана "
    f"найкоротший шлях {' -> '.join(path)} довжиною {shortest_len:.0f} збігається з результатом повного перебору "
    f"всіх можливих шляхів; " + ("для цього конкретного набору ваг він також збігся з жадібним вибором "
    "найдешевшого ребра на кожному кроці, проте правило вибору однаково лишається «вартість кроку + V(w) наступної "
    "вершини», а не вартість кроку окремо." if greedy_matches else
    "на відміну від жадібного вибору найдешевшого ребра на кожному кроці (який дав гірший результат), ДП обирає "
    "крок за сумою його вартості й функції Беллмана наступної вершини."),
    f"Додаткова задача заміни обладнання на горизонті {HORIZON} років підтвердила, що принцип Беллмана однаково "
    f"добре працює для задач максимізації прибутку незалежно від конкретного змісту стану (ресурс, вершина мережі "
    f"чи вік обладнання) — оптимальний сумарний прибуток {total_eq:.0f} теж підтверджений повним перебором.",
    "Динамічне програмування звело задачу комбінаторного вибору (що зростає як O(R^N)) до послідовності з N "
    "простих одновимірних підзадач (O(N·R²)), не втративши точності розв'язку — у всіх трьох задачах роботи "
    "результат ДП точно збігся з повним перебором.",
    "Головне обмеження методу — прокляття розмірності: складність зростає експоненційно вже не за N (кількістю "
    "кроків), а за розмірністю вектора стану; доки стан можна описати одним-двома числами (як у цій роботі), ДП "
    "залишається практичним і точним інструментом, але для векторного стану високої розмірності потрібні "
    "наближені варіанти методу.",
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
print(f"Готово: {docx_path} (+ {network_path}, {complexity_path})")
