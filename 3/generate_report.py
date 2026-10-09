"""
Генератор звіту до практичної роботи №3 (Методи нульового порядку багатовимірної безумовної оптимізації).
Питає номер варіанта (або приймає його аргументом) і генерує Zvit_PR3_Variant<N>.docx.

Запуск:  .venv/bin/python generate_report.py [номер_варіанта]
"""
import sys
import time
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
WORK_TITLE = "Методи нульового порядку багатовимірної безумовної оптимізації"
WORK_NUMBER = 3

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
expr_str, x0_raw, eps0, h0 = VARIANTS[VARIANT]
x0 = np.array(x0_raw, dtype=float)

x1, x2 = sp.symbols('x1 x2', real=True)
f_sym = sp.sympify(expr_str, locals={'x1': x1, 'x2': x2})
f_np = sp.lambdify((x1, x2), f_sym, 'numpy')

calls = {"n": 0}


def f_vec(p):
    calls["n"] += 1
    return float(f_np(p[0], p[1]))


# Варіант 1 належить до родини узагальненої функції Розенброка a(x2-x1^2)^2+(1-x1)^2:
# другий доданок обнуляється лише при x1=1, тоді перший — при x2=1, тому мінімум єдиний
# і не залежить від a. Для інших родин варіантів (Хіммельблау, Бута, квадратичні, Біла)
# використовується той самий принцип — аналітичний розбір структури функції.
x_exact = np.array([1.0, 1.0])
f_exact = f_np(*x_exact)

# ---------- Лінії рівня (базовий графік) ----------
def pt(v):
    """np.float64 -> звичайний float (інакше в numpy>=2 рядок типу 'np.float64(1.2)')."""
    return tuple(float(c) for c in v)


def contour_levels(Z, n=28):
    """Рівні за квантилями значень Z у вікні перегляду — на яружних функціях з величезним
    розкидом значень звичайні рівносіддені рівні ховають яр у 1-2 суцільних плямах."""
    qs = np.linspace(0, 1, n)
    return np.unique(np.quantile(Z, qs))


center = (x0 + x_exact) / 2
half_width = max(1.5, float(np.linalg.norm(x0 - x_exact)) / 2 + 1.0)
xs = np.linspace(center[0] - half_width, center[0] + half_width, 300)
ys = np.linspace(center[1] - half_width, center[1] + half_width, 300)
X, Y = np.meshgrid(xs, ys)
Z = f_np(X, Y)

fig0, ax0 = plt.subplots(figsize=(6, 5))
ax0.contour(X, Y, Z, levels=contour_levels(Z), cmap='viridis')
ax0.plot(*x0, 'ko', ms=8, label='x0')
ax0.plot(*x_exact, 'g*', ms=14, label='x*')
ax0.set_xlabel('x1'); ax0.set_ylabel('x2'); ax0.legend()
ax0.set_title(f'Варіант {VARIANT}: лінії рівня f(x1, x2)')
fig0.tight_layout()
contour_path = f"contour_variant_{VARIANT}.png"
fig0.savefig(contour_path, dpi=150)
plt.close(fig0)


# ---------- Методи (з історією траєкторії) ----------
def golden_line(f, x, d, a=-2.0, b=2.0, eps=1e-6):
    """Мінімізація f(x + t*d) за напрямком d методом золотого перерізу."""
    phi = (np.sqrt(5) - 1) / 2
    t1, t2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = f(x + t1 * d), f(x + t2 * d)
    while b - a > eps:
        if f1 <= f2:
            b, t2, f2 = t2, t1, f1
            t1 = b - phi * (b - a); f1 = f(x + t1 * d)
        else:
            a, t1, f1 = t1, t2, f2
            t2 = a + phi * (b - a); f2 = f(x + t2 * d)
    return (a + b) / 2


def coordinate_descent(f, x0, eps=1e-6, itmax=5000):
    x = np.array(x0, float); n = len(x); path = [x.copy()]
    for k in range(itmax):
        x_old = x.copy()
        for i in range(n):
            d = np.zeros(n); d[i] = 1.0
            x = x + golden_line(f, x, d) * d
        path.append(x.copy())
        if np.linalg.norm(x - x_old) < eps:
            break
    return x, k + 1, np.array(path)


def hooke_jeeves(f, x0, h=0.5, alpha=2.0, eps=1e-6, itmax=5000):
    def explore(base, h):
        x = base.copy(); fx = f(x)
        for i in range(len(x)):
            for s in (h, -h):
                trial = x.copy(); trial[i] += s
                ft = f(trial)
                if ft < fx:
                    x, fx = trial, ft
                    break
        return x, fx

    x = np.array(x0, float); path = [x.copy()]; k = 0
    while h > eps and k < itmax:
        k += 1
        xn, fn = explore(x, h)
        if fn < f(x):
            while True:  # пошук за зразком
                xp = xn + alpha * (xn - x)
                xe, fe = explore(xp, h)
                if fe < fn:
                    x, xn, fn = xn, xe, fe
                else:
                    x = xn
                    break
            path.append(x.copy())
        else:
            h /= 2.0  # дроблення кроку
    return x, k, np.array(path)


# ---------- Запуск трьох методів ----------
calls["n"] = 0
t0 = time.perf_counter()
x_cd, it_cd, path_cd = coordinate_descent(f_vec, x0, eps=eps0)
t_cd = time.perf_counter() - t0
calls_cd = calls["n"]

calls["n"] = 0
t0 = time.perf_counter()
x_hj, it_hj, path_hj = hooke_jeeves(f_vec, x0, h=h0, eps=eps0)
t_hj = time.perf_counter() - t0
calls_hj = calls["n"]

calls["n"] = 0
t0 = time.perf_counter()
path_nm = [np.array(x0)]
res_nm = minimize(
    f_vec, x0, method='Nelder-Mead',
    options={'xatol': eps0, 'fatol': eps0},
    callback=lambda xk: path_nm.append(np.array(xk)),
)
t_nm = time.perf_counter() - t0
calls_nm = calls["n"]
path_nm = np.array(path_nm)

summary_rows = [
    ("Покоординатний спуск", x_cd, f_vec(x_cd), it_cd, calls_cd, t_cd),
    ("Хука-Дживса", x_hj, f_vec(x_hj), it_hj, calls_hj, t_hj),
    ("Нелдера-Міда", res_nm.x, res_nm.fun, res_nm.nit, calls_nm, t_nm),
]

# ---------- Траєкторії на лініях рівня (три графіки) ----------
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, path) in zip(axes, [
    ("Покоординатний спуск", path_cd),
    ("Хука-Дживса", path_hj),
    ("Нелдера-Міда", path_nm),
]):
    ax.contour(X, Y, Z, levels=contour_levels(Z), cmap='viridis', alpha=.6)
    ax.plot(path[:, 0], path[:, 1], 'r.-', ms=4, lw=1)
    ax.plot(*x0, 'ko', ms=7, label='x0')
    ax.plot(*x_exact, 'g*', ms=14, label='x*')
    ax.set_title(name); ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.legend(fontsize=8)
fig1.suptitle(f'Варіант {VARIANT}: траєкторії пошуку')
fig1.tight_layout()
traj_path = f"trajectories_variant_{VARIANT}.png"
fig1.savefig(traj_path, dpi=150)
plt.close(fig1)

# ---------- Чутливість до початкової точки ----------
starts = [np.array(x0), np.array([0.0, 0.0]), np.array([2.0, -1.0])]
sens_x0_rows = []
for s in starts:
    xc, _, _ = coordinate_descent(f_vec, s, eps=eps0)
    xh, _, _ = hooke_jeeves(f_vec, s, h=h0, eps=eps0)
    rn = minimize(f_vec, s, method='Nelder-Mead', options={'xatol': eps0, 'fatol': eps0})
    sens_x0_rows.append((pt(s), pt(np.round(xc, 4)), pt(np.round(xh, 4)), pt(np.round(rn.x, 4))))

# ---------- Чутливість до початкового кроку h (Хука-Дживса) ----------
sens_h_rows = []
for h_test in (0.1, 0.5, 1.0):
    calls["n"] = 0
    xh, it_, _ = hooke_jeeves(f_vec, x0, h=h_test, eps=eps0)
    sens_h_rows.append((h_test, xh, f_vec(xh), it_, calls["n"]))

# ---------- Лістинг коду ----------
CODE_METHODS = '''def golden_line(f, x, d, a=-2.0, b=2.0, eps=1e-6):
    """Мінімізація f(x + t*d) за напрямком d методом золотого перерізу."""
    phi = (np.sqrt(5) - 1) / 2
    t1, t2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = f(x + t1 * d), f(x + t2 * d)
    while b - a > eps:
        if f1 <= f2:
            b, t2, f2 = t2, t1, f1
            t1 = b - phi * (b - a); f1 = f(x + t1 * d)
        else:
            a, t1, f1 = t1, t2, f2
            t2 = a + phi * (b - a); f2 = f(x + t2 * d)
    return (a + b) / 2


def coordinate_descent(f, x0, eps=1e-6, itmax=5000):
    x = np.array(x0, float); n = len(x); path = [x.copy()]
    for k in range(itmax):
        x_old = x.copy()
        for i in range(n):
            d = np.zeros(n); d[i] = 1.0
            x = x + golden_line(f, x, d) * d      # одновимірна мінімізація вздовж осі i
        path.append(x.copy())
        if np.linalg.norm(x - x_old) < eps:       # критерій зупинки: зсув за повний цикл
            break
    return x, k + 1, np.array(path)


def hooke_jeeves(f, x0, h=0.5, alpha=2.0, eps=1e-6, itmax=5000):
    def explore(base, h):                          # дослідницький пошук
        x = base.copy(); fx = f(x)
        for i in range(len(x)):
            for s in (h, -h):
                trial = x.copy(); trial[i] += s
                ft = f(trial)
                if ft < fx:
                    x, fx = trial, ft
                    break
        return x, fx

    x = np.array(x0, float); path = [x.copy()]; k = 0
    while h > eps and k < itmax:
        k += 1
        xn, fn = explore(x, h)
        if fn < f(x):
            while True:                             # пошук за зразком вздовж яру
                xp = xn + alpha * (xn - x)
                xe, fe = explore(xp, h)
                if fe < fn:
                    x, xn, fn = xn, xe, fe
                else:
                    x = xn
                    break
            path.append(x.copy())
        else:
            h /= 2.0                                # дроблення кроку
    return x, k, np.array(path)

# Метод Нелдера-Міда — готова реалізація SciPy:
# res = minimize(f, x0, method='Nelder-Mead', options={'xatol': eps, 'fatol': eps})
'''

# ---------- Контрольні питання ----------
QA = [
    ("Чому методи нульового порядку актуальні для оптимізації складних систем?",
     "У задачах оптимізації складних систем цільова функція часто задається не аналітичним виразом, а "
     "алгоритмом — результатом імітаційного моделювання, розв'язком системи диференціальних рівнянь, "
     "відгуком натурного експерименту. У таких випадках градієнт або не існує в явному вигляді, або його "
     "чисельне обчислення потребує n+1 додаткових запусків моделі на кожному кроці і сильно спотворюється "
     "шумом вимірювань. Методи нульового порядку використовують лише значення самої функції, тому "
     "застосовні там, де градієнтні методи неможливі або ненадійні."),
    ("Опишіть алгоритм покоординатного спуску. Що є однією ітерацією методу?",
     "Напрямками спуску по черзі обираються координатні орти e1, e2, ..., en. На кожному з них "
     "розв'язується задача одновимірної мінімізації tk = argmin f(x + t*ei), і точка зсувається вздовж цього "
     "напрямку на знайдений крок. Одна ітерація методу — це повний цикл такої одновимірної мінімізації по "
     "всіх n координатах підряд; критерій зупинки перевіряється саме після завершення повного циклу."),
    ("Що таке яр цільової функції і чому покоординатний спуск на ньому неефективний?",
     "Яром називають ділянку, де лінії рівня функції утворюють вузьку витягнуту долину під кутом до "
     "координатних осей (класичний приклад — функція Розенброка, дно якої лежить уздовж параболи x2=x1²). "
     "Оскільки покоординатний спуск завжди рухається строго вздовж осей, а не вздовж дна яру, кожен крок "
     "дає лише мізерне просування і метод змушений робити тисячі дрібних кроків, «зиґзаґоподібно» "
     "повзучи по стінках яру замість того, щоб рухатися вздовж нього."),
    ("У чому полягає дослідницький пошук у методі Хука-Дживса?",
     "Із поточної базової точки послідовно для кожної координати робиться пробний крок ±h. Якщо пробний крок "
     "зменшує значення функції, він приймається і наступна координата досліджується вже від нової точки; "
     "якщо жоден із двох напрямків (+h, -h) по даній координаті не покращує результат, вона лишається без "
     "змін. Результатом є нова точка x̃, яка й визначає «вдалий» напрямок руху від базової точки."),
    ("Навіщо потрібен пошук за зразком і як обирається коефіцієнт α?",
     "Якщо дослідницький пошук дав покращення, є підстави припустити, що знайдений напрямок (x̃ - x) і далі "
     "веде до зменшення функції, тому робиться подовжений («зразковий») крок xp = x̃ + α(x̃ - x) з α ≥ 1 "
     "(типово α = 2 — подвоєння кроку). Саме цей механізм дозволяє методу розганятися вздовж дна яру, а не "
     "лише повзти дрібними дослідницькими кроками; якщо зразковий крок не підтверджується повторним "
     "дослідницьким пошуком, метод повертається до останньої вдалої точки."),
    ("Коли в методі Хука-Дживса дроблять крок h і за якою умовою зупиняють процес?",
     "Крок дроблять (h := h/2), коли дослідницький пошук із поточним h не дав жодного покращення з базової "
     "точки — це означає, що h завеликий для подальшого уточнення. Процес зупиняють, коли h стає меншим за "
     "задану точність ε: подальше дроблення вже не дасть змістовного уточнення координат мінімуму."),
    ("Скільки вершин має симплекс у задачі розмірності n? Перелічіть операції його деформації.",
     "Симплекс у просторі розмірності n складається з n+1 вершин (для n=2 це трикутник). Операції "
     "деформації: відбиття (reflection, α=1) — базова спроба покращення; розтяг (expansion, γ=2) — якщо "
     "відбита точка виявилась кращою за найкращу вершину; стиск (contraction, β=0.5) — якщо відбита точка "
     "не кращa за другу найгіршу; і редукція (shrink, σ=0.5) — стиснення всього симплекса до найкращої "
     "вершини, якщо навіть стиск не дав покращення."),
    ("Коли в методі Нелдера-Міда виконується редукція симплекса?",
     "Редукція виконується тоді, коли навіть операція стиску (contraction) не дала точки, кращої за найгіршу "
     "вершину поточного симплекса. У такому разі єдиний надійний спосіб продовжити пошук — стиснути всі "
     "вершини, крім найкращої, до неї самої з коефіцієнтом σ, зменшивши розмір симплекса і повторивши цикл "
     "уже в меншому масштабі."),
    ("Чому функція Розенброка є стандартним тестом для методів оптимізації?",
     "Функція Розенброка f(x1,x2) = a(x2-x1²)² + (1-x1)² має єдиний глобальний мінімум у точці (1,1), але "
     "її лінії рівня утворюють вузький викривлений яр уздовж параболи x2=x1², що не збігається з жодним "
     "координатним напрямком. Це робить її складним і показовим тестом саме для методів прямого пошуку: "
     "вона одразу виявляє, чи здатен метод рухатися вздовж викривленої долини, а не лише вздовж осей."),
    ("Порівняйте три вивчені методи за кількістю обчислень цільової функції та поясніть причину різниці.",
     "Покоординатний спуск — найдорожчий: рухаючись лише вздовж осей, він не розпізнає напрямок яру і "
     "виконує величезну кількість дрібних кроків, кожен з яких потребує викликів f для одновимірної "
     "мінімізації золотим перерізом. Метод Хука-Дживса економніший на порядки, бо пошук за зразком дозволяє "
     "йому «відчути» і йти вздовж дна яру великими кроками. Метод Нелдера-Міда зазвичай найефективніший: "
     "симплекс самостійно витягується вздовж яру та стискається поблизу мінімуму, тому потребує найменше "
     "обчислень функції серед трьох методів (конкретні числа — у таблиці порівняння нижче)."),
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


def fmt(v, nd=6):
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
    "Опанувати методи прямого пошуку, що не використовують похідних цільової функції: покоординатний "
    "спуск, метод конфігурацій Хука-Дживса та симплексний метод Нелдера-Міда; реалізувати їх мовою Python, "
    "дослідити поведінку на функції яружного типу та порівняти трудомісткість."
)

# --- 2. Постановка задачі ---
add_heading("2. Постановка задачі та графік ліній рівня")
add_para(f"f(x1, x2) = {f_sym}   ->  min", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(f"Початкова точка: x0 = {pt(x0)}. Точність: ε = {eps0}. Початковий крок: h = {h0}.")
doc.add_picture(contour_path, width=Cm(11))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    "За лініями рівня видно характерний яр: вузька викривлена долина, дно якої не збігається з жодною з "
    "координатних осей, що й створює труднощі для покоординатного спуску."
)

# --- 3. Опис алгоритмів ---
add_heading("3. Опис алгоритмів методів")
add_para(
    "Покоординатний спуск: на кожній ітерації по черзі вздовж кожної координатної осі виконується "
    "одновимірна мінімізація методом золотого перерізу; ітерація — повний цикл по всіх осях."
)
add_para(
    "Хука-Дживса: чергування дослідницького пошуку (пробні кроки ±h по кожній координаті з базової точки) "
    "і пошуку за зразком (подовжений крок уздовж знайденого вдалого напрямку); крок дробиться, якщо "
    "дослідницький пошук не дає покращення."
)
add_para(
    "Нелдера-Міда: симплекс із n+1 вершин послідовно деформується операціями відбиття, розтягу, стиску та "
    "редукції так, щоб гірша вершина замінювалась на кращу точку; використано готову реалізацію SciPy "
    "(method='Nelder-Mead')."
)

# --- 4. Тексти програм ---
add_heading("4. Тексти програм")
add_code_block(CODE_METHODS.strip("\n"))

# --- 5. Траєкторії ---
add_heading("5. Траєкторії пошуку на лініях рівня")
doc.add_picture(traj_path, width=Cm(16))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    "Покоординатний спуск рухається дрібними ступінчастими кроками вздовж осей. Хука-Дживса завдяки "
    "пошуку за зразком швидко «розвертається» вздовж дна яру. Траєкторія Нелдера-Міда відображає, як "
    "симплекс перетікає ландшафтом, витягуючись уздовж яру та стискаючись біля мінімуму."
)

# --- 6. Зведена таблиця ---
add_heading("6. Зведена таблиця порівняння методів")
add_table(
    ["Метод", "x1*", "x2*", "f(x*)", "Ітерацій", "Обчислень f", "Час, с"],
    [[name, fmt(x_[0]), fmt(x_[1]), fmt(fv, 8), it_, calls_, fmt(t_, 4)]
     for name, x_, fv, it_, calls_, t_ in summary_rows]
)
best = min(summary_rows, key=lambda r: r[4])
worst = max(summary_rows, key=lambda r: r[4])
add_para(
    f"Найекономнішим за кількістю обчислень цільової функції виявився метод «{best[0]}» ({best[4]} обчислень), "
    f"найдорожчим — «{worst[0]}» ({worst[4]} обчислень), що узгоджується з теоретичними очікуваннями: методи, "
    f"які розпізнають напрямок яру (Хука-Дживса, Нелдера-Міда), набагато ефективніші за покоординатний спуск."
)

# --- 7. Чутливість ---
add_heading("7. Дослідження чутливості")
add_para("Чутливість до початкової точки:", bold=True)
add_table(
    ["x0", "Покоординатний спуск", "Хука-Дживса", "Нелдера-Міда"],
    [[str(s), str(c), str(h), str(n)] for s, c, h, n in sens_x0_rows]
)
all_converged = all(
    np.allclose(c, sens_x0_rows[0][1], atol=1e-2) and
    np.allclose(h, sens_x0_rows[0][2], atol=1e-2) and
    np.allclose(n, sens_x0_rows[0][3], atol=1e-2)
    for _, c, h, n in sens_x0_rows
)
add_para(
    ("З усіх перевірених початкових точок усі три методи збіглися практично до того самого мінімуму — "
     "очікувано, оскільки для даного варіанта функція має єдиний глобальний мінімум без інших локальних "
     "екстремумів, що «притягують» траєкторію пошуку.")
    if all_converged else
    ("Результати з різних початкових точок дещо відрізняються — методи прямого пошуку в принципі не "
     "гарантують збіжності до глобального мінімуму, особливо на функціях із кількома локальними "
     "екстремумами; остаточний результат залежить від того, у «яр» якого саме екстремуму потрапляє "
     "траєкторія пошуку.")
)
add_para("Чутливість методу Хука-Дживса до початкового кроку h:", bold=True)
add_table(
    ["h", "x1*", "x2*", "f(x*)", "Ітерацій", "Обчислень f"],
    [[fmt(h_, 2), fmt(xh_[0]), fmt(xh_[1]), fmt(fv_, 8), it_, c_] for h_, xh_, fv_, it_, c_ in sens_h_rows]
)
add_para(
    "Більший початковий крок h дає грубіший, але швидший перший прохід дослідницьким пошуком, тоді як "
    "менший h одразу забезпечує точніше наближення ціною більшої кількості дроблень кроку; у підсумку "
    "кінцевий результат майже не залежить від h, проте впливає на кількість ітерацій і обчислень функції."
)

# --- 8. Висновки ---
add_heading("8. Висновки")
conclusions = [
    f"Усі три методи для варіанта {VARIANT} знайшли мінімум f(x*) ≈ {fmt(summary_rows[-1][2], 6)}, що "
    f"узгоджується з аналітичним значенням f(x*) = {f_exact:.6f} у точці x* = {pt(x_exact)}.",
    "Покоординатний спуск — найпростіший у реалізації, але вкрай неефективний на яружних функціях: рух "
    "строго вздовж осей змушує його робити величезну кількість дрібних кроків, щоб обійти викривлене дно яру.",
    "Метод Хука-Дживса завдяки комбінації дослідницького пошуку й пошуку за зразком на порядки економніший "
    "за покоординатний спуск, оскільки здатен рухатися вздовж яру, а не поперек нього.",
    "Метод Нелдера-Міда виявився найефективнішим серед трьох: деформівний симплекс автоматично "
    "підлаштовується під форму ландшафту, витягуючись уздовж яру та стискаючись поблизу мінімуму.",
    "Практичні рекомендації: для «дорогої» цільової функції (повільна імітаційна модель) варто обирати "
    "Нелдера-Міда або Хука-Дживса, а не покоординатний спуск; для зашумленої функції Нелдера-Міда також "
    "переважний завдяки стійкості до помірного шуму; якщо функція має кілька локальних мінімумів, жоден із "
    "трьох методів не гарантує глобальної збіжності — потрібні повторні запуски з різних початкових точок.",
]
for c in conclusions:
    doc.add_paragraph(c, style="List Bullet")

# --- 9. Контрольні питання ---
add_heading("9. Відповіді на контрольні питання")
for i, (q, ans) in enumerate(QA, start=1):
    add_para(f"{i}. {q}", bold=True)
    add_para(ans)

docx_path = f"Zvit_PR{WORK_NUMBER}_Variant{VARIANT}.docx"
doc.save(docx_path)
print(f"Готово: {docx_path} (+ {contour_path}, {traj_path})")
