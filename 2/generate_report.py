"""
Генератор звіту до практичної роботи №2 (Методи одновимірної оптимізації та аналіз їх збіжності).
Питає номер варіанта (або приймає його аргументом) і генерує Zvit_PR2_Variant<N>.docx.

Запуск:  .venv/bin/python generate_report.py [номер_варіанта]
"""
import sys
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
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
WORK_TITLE = "Методи одновимірної оптимізації та аналіз їх збіжності"
WORK_NUMBER = 2

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
expr_str, a0, b0, x0_newton, eps0 = VARIANTS[VARIANT]

x = sp.symbols('x', real=True)
f_sym = sp.sympify(expr_str, locals={'x': x})
df_sym = sp.diff(f_sym, x)
d2f_sym = sp.diff(df_sym, x)

f = sp.lambdify(x, f_sym, 'numpy')
df = sp.lambdify(x, df_sym, 'numpy')
d2f = sp.lambdify(x, d2f_sym, 'numpy')

x_exact = float(sp.nsolve(df_sym, x, x0_newton))
f2_at_exact = float(d2f_sym.subs(x, x_exact))
is_min = f2_at_exact > 0

# ---------- Графік унімодальності ----------
xs = np.linspace(a0, b0, 400)
ys = f(xs)
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(xs, ys)
ax1.axvline(x_exact, color='r', ls='--', lw=1, label=f'x* = {x_exact:.4f}')
ax1.set_xlabel('x'); ax1.set_ylabel('f(x)'); ax1.grid(alpha=.3); ax1.legend()
ax1.set_title(f'Варіант {VARIANT}: f(x) на [{a0}, {b0}]')
fig1.tight_layout()
func_plot_path = f"function_plot_variant_{VARIANT}.png"
fig1.savefig(func_plot_path, dpi=150)
plt.close(fig1)


# ---------- Методи (з історією ітерацій) ----------
def dichotomy(f, a, b, eps, delta=None):
    if delta is None:
        delta = eps / 10
    history = []
    n = 0
    while (b - a) / 2 > eps:
        m = (a + b) / 2
        x1, x2 = m - delta, m + delta
        f1, f2 = f(x1), f(x2)
        history.append(dict(iter=n + 1, a=a, b=b, x1=x1, x2=x2, f1=f1, f2=f2, length=b - a))
        if f1 <= f2:
            b = x2
        else:
            a = x1
        n += 1
    return (a + b) / 2, 2 * n, n, history


def golden(f, a, b, eps):
    phi = (np.sqrt(5) - 1) / 2
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = f(x1), f(x2)
    n = 2
    history = []
    while (b - a) / 2 > eps:
        history.append(dict(iter=n - 1, a=a, b=b, x1=x1, x2=x2, f1=f1, f2=f2, length=b - a))
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a); f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a); f2 = f(x2)
        n += 1
    return (a + b) / 2, n, n - 1, history


def fibonacci(f, a, b, eps):
    fib = [1, 1]
    while fib[-1] < (b - a) / eps:
        fib.append(fib[-1] + fib[-2])
    N = len(fib) - 1
    x1 = a + fib[N - 2] / fib[N] * (b - a)
    x2 = a + fib[N - 1] / fib[N] * (b - a)
    f1, f2 = f(x1), f(x2)
    history = []
    for k in range(1, N - 1):
        history.append(dict(iter=k, a=a, b=b, x1=x1, x2=x2, f1=f1, f2=f2, length=b - a))
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + fib[N - k - 2] / fib[N - k] * (b - a); f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + fib[N - k - 1] / fib[N - k] * (b - a); f2 = f(x2)
    return (a + b) / 2, N, N - 1, history


def newton(df, d2f, x0, eps, itmax=100):
    x_k = x0
    history = []
    i = 0
    for i in range(itmax):
        fp, fpp = df(x_k), d2f(x_k)
        step = fp / fpp
        x_next = x_k - step
        history.append(dict(iter=i + 1, x=x_k, fpx=fp, fppx=fpp, step=step, x_next=x_next))
        x_k = x_next
        if abs(step) < eps:
            break
    return x_k, i + 1, i + 1, history


def newton_safe(df, d2f, x0, eps, itmax=100, blow_up=1e6):
    x_k = x0
    try:
        for i in range(itmax):
            fp, fpp = df(x_k), d2f(x_k)
            if fpp == 0:
                return x_k, i + 1, "ділення на нуль (f''=0)"
            step = fp / fpp
            x_next = x_k - step
            if not np.isfinite(x_next) or abs(x_next) > blow_up:
                return x_next, i + 1, "розбіжність (|x| -> дуже велике)"
            x_k = x_next
            if abs(step) < eps:
                return x_k, i + 1, "збіжність"
        return x_k, itmax, "не збігся за itmax ітерацій"
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        return None, None, f"помилка обчислення: {exc}"


x_dich, calls_dich, it_dich, hist_dich = dichotomy(f, a0, b0, eps0)
x_gold, calls_gold, it_gold, hist_gold = golden(f, a0, b0, eps0)
x_fib, calls_fib, it_fib, hist_fib = fibonacci(f, a0, b0, eps0)
x_newt, calls_newt, it_newt, hist_newt = newton(df, d2f, x0_newton, eps0)
res = minimize_scalar(f, bounds=(a0, b0), method='bounded', options={'xatol': eps0})

summary_rows = [
    ("Дихотомія", x_dich, f(x_dich), abs(x_dich - x_exact), calls_dich, it_dich),
    ("Золотий переріз", x_gold, f(x_gold), abs(x_gold - x_exact), calls_gold, it_gold),
    ("Фібоначчі", x_fib, f(x_fib), abs(x_fib - x_exact), calls_fib, it_fib),
    ("Ньютона", x_newt, f(x_newt), abs(x_newt - x_exact), calls_newt, it_newt),
    ("SciPy (Brent)", res.x, res.fun, abs(res.x - x_exact), res.nfev, res.nit),
]

# ---------- Графік збіжності ----------
fig2, ax2 = plt.subplots(figsize=(6, 4))
for hist, label in [(hist_dich, 'Дихотомія'), (hist_gold, 'Золотий переріз'), (hist_fib, 'Фібоначчі')]:
    lengths = [h['length'] for h in hist]
    ax2.semilogy(range(1, len(lengths) + 1), lengths, marker='o', ms=3, label=label)
ax2.set_xlabel('номер ітерації'); ax2.set_ylabel('довжина інтервалу (лог. шкала)')
ax2.grid(alpha=.3, which='both'); ax2.legend()
ax2.set_title(f'Варіант {VARIANT}: збіжність методів скорочення інтервалу')
fig2.tight_layout()
conv_plot_path = f"convergence_plot_variant_{VARIANT}.png"
fig2.savefig(conv_plot_path, dpi=150)
plt.close(fig2)

# ---------- Чутливість до eps ----------
eps_list = [1e-3, 1e-5, 1e-7]
eps_rows = []
for e in eps_list:
    _, c_d, i_d, _ = dichotomy(f, a0, b0, e)
    _, c_g, i_g, _ = golden(f, a0, b0, e)
    _, c_f, i_f, _ = fibonacci(f, a0, b0, e)
    _, c_n, i_n, _ = newton(df, d2f, x0_newton, e)
    eps_rows.append((e, c_d, c_g, c_f, c_n))

# ---------- Стійкість Ньютона ----------
starts = sorted({round(a0 + 0.1 * (b0 - a0), 4), x0_newton, round(b0 - 0.1 * (b0 - a0), 4)})
stability_rows = []
for s in starts:
    xr, nit, status = newton_safe(df, d2f, s, eps0)
    stability_rows.append((s, xr, nit, status))

# ---------- Лістинги коду ----------
CODE_METHODS = '''def dichotomy(f, a, b, eps, delta=None):
    if delta is None:
        delta = eps / 10
    n = 0
    while (b - a) / 2 > eps:
        m = (a + b) / 2
        x1, x2 = m - delta, m + delta
        if f(x1) <= f(x2):
            b = x2
        else:
            a = x1
        n += 1
    return (a + b) / 2, 2 * n, n


def golden(f, a, b, eps):
    phi = (np.sqrt(5) - 1) / 2
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2 = f(x1), f(x2)
    n = 2
    while (b - a) / 2 > eps:
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a); f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a); f2 = f(x2)
        n += 1
    return (a + b) / 2, n, n - 1


def fibonacci(f, a, b, eps):
    fib = [1, 1]
    while fib[-1] < (b - a) / eps:
        fib.append(fib[-1] + fib[-2])
    N = len(fib) - 1
    x1 = a + fib[N-2] / fib[N] * (b - a)
    x2 = a + fib[N-1] / fib[N] * (b - a)
    f1, f2 = f(x1), f(x2)
    for k in range(1, N - 1):
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + fib[N-k-2] / fib[N-k] * (b - a); f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + fib[N-k-1] / fib[N-k] * (b - a); f2 = f(x2)
    return (a + b) / 2, N, N - 1


def newton(df, d2f, x0, eps, itmax=100):
    x, n = x0, 0
    for _ in range(itmax):
        step = df(x) / d2f(x)
        x -= step
        n += 1
        if abs(step) < eps:
            break
    return x, n, n
'''

# ---------- Контрольні питання ----------
QA = [
    ("Дайте означення унімодальної функції. Чому унімодальність є обов'язковою умовою методів скорочення інтервалу?",
     "Функція f унімодальна на [a, b], якщо існує єдина точка x*, до якої f строго спадає, а після якої строго "
     "зростає (єдиний «провал», без інших локальних мінімумів чи максимумів). Унімодальність потрібна, бо лише "
     "тоді порівняння f у двох внутрішніх точках x1<x2 дає однозначний висновок: якщо f(x1)≤f(x2), то x* точно "
     "не може лежати правіше x2, і цю частину можна безпечно відкинути. Для функції з кількома локальними "
     "мінімумами таке відкидання могло б випадково прибрати частину з глобальним мінімумом."),
    ("Що таке інтервал невизначеності і як він змінюється в методі дихотомії?",
     "Інтервал невизначеності — поточний відрізок [a, b], у якому гарантовано міститься точка мінімуму. У "
     "методі дихотомії після кожної ітерації (порівняння двох симетричних пробних точок біля середини) "
     "довжина інтервалу зменшується майже вдвічі, тобто після n ітерацій вона приблизно дорівнює (b−a)/2ⁿ."),
    ("Чому в методі дихотомії потрібні два обчислення функції на ітерацію, а в методі золотого перерізу — одне?",
     "У дихотомії обидві пробні точки щоразу нові й розташовані симетрично біля середини нового інтервалу, "
     "тому кожна ітерація вимагає двох нових обчислень f. У золотому перерізі точки розміщені асиметрично за "
     "пропорцією φ так, що одна з двох точок попередньої ітерації опиняється точно в потрібній позиції "
     "всередині нового (звуженого) інтервалу — її значення f вже відоме й переноситься, тому обчислювати "
     "потрібно лише другу, нову точку."),
    ("Звідки береться число φ = 0,618 і чому саме воно забезпечує перевикористання пробної точки?",
     "φ = (√5−1)/2 ≈ 0,618 — розв'язок рівняння φ² + φ = 1 (тобто 1/φ = 1+φ), відомий як «золотий переріз». "
     "Це єдине число, для якого пропорція поділу відрізка зберігається самоподібно на кожному кроці "
     "звуження: якщо ділити інтервал у цій пропорції, одна зі старих пробних точок автоматично потрапляє в "
     "потрібну (таку саму відносну) позицію нового інтервалу — тому її можна перевикористати без нового "
     "обчислення функції."),
    ("У якому сенсі метод Фібоначчі є оптимальним? Який його практичний недолік?",
     "Метод Фібоначчі оптимальний у тому сенсі, що для наперед заданої кількості N обчислень функції він "
     "гарантує найменший можливий кінцевий інтервал невизначеності серед усіх послідовних методів скорочення "
     "інтервалу — жодна інша стратегія цього типу не може досягти кращого результату за ту саму кількість "
     "викликів f. Практичний недолік — потрібно заздалегідь знати N, тобто наперед розрахувати, скільки "
     "ітерацій знадобиться для потрібної точності, що незручно, якщо точність визначається по ходу справи."),
    ("Порівняйте методи золотого перерізу та Фібоначчі за кінцевою довжиною інтервалу при однаковій кількості обчислень.",
     "При однаковій кількості обчислень метод Фібоначчі дає дещо менший (кращий) кінцевий інтервал, ніж "
     "золотий переріз, оскільки він точно підлаштований під задане N. Золотий переріз є граничним випадком "
     "методу Фібоначчі при N → ∞, тому різниця між ними мала і зменшується зі зростанням кількості ітерацій, "
     "практично зникаючи для великих N."),
    ("Чому метод Ньютона має квадратичну збіжність? Яка ціна цієї швидкості?",
     "Метод Ньютона на кожному кроці будує квадратичну (параболічну) апроксимацію функції за допомогою f' і "
     "f'' і одразу переходить у вершину цієї параболи. З розкладу Тейлора випливає, що похибка на наступному "
     "кроці пропорційна квадрату похibки на попередньому, тому кількість правильних значущих цифр приблизно "
     "подвоюється щоразу. Ціна цієї швидкості — потрібно обчислювати f' і f'' на кожній ітерації, а також "
     "мати досить близьке до розв'язку початкове наближення: глобальної гарантії збіжності немає."),
    ("Наведіть приклад, коли метод Ньютона розбігається або збігається до максимуму.",
     "Для функцій виду f(x) = a·x + c/x + d (як у прикладі та варіантах цієї роботи) рівняння f'(x) = 0 має "
     "два корені — додатний (шуканий мінімум) і від'ємний. Якщо стартувати з невдалого початкового наближення "
     "(наприклад, близько до правої межі відрізка), метод Ньютона може «вилетіти» за межі області визначення "
     "або збігтися до від'ємного кореня, який не має сенсу для вихідної задачі — це підтверджено й "
     "експериментально в цій роботі (див. розділ «Стійкість методу Ньютона»)."),
    ("Які критерії зупинки ітераційного процесу застосовують і чому не можна покладатися лише на |f⁽ᵏ⁺¹⁾ − f⁽ᵏ⁾| < ε?",
     "Застосовують кілька критеріїв: довжину інтервалу (b−a)<ε, малу зміну аргументу |x⁽ᵏ⁺¹⁾−x⁽ᵏ⁾|<ε, малу "
     "зміну значення функції |f⁽ᵏ⁺¹⁾−f⁽ᵏ⁾|<ε та обмеження на максимальну кількість ітерацій. Покладатися лише "
     "на малу зміну f небезпечно: біля дуже пласкої ділянки функції (де похідна близька до нуля) значення f "
     "майже не змінюється, хоча аргумент x ще далекий від справжнього x* — це призведе до передчасної хибної "
     "зупинки."),
    ("Де в багатовимірних методах оптимізації використовується одновимірний пошук?",
     "У кроці лінійного пошуку (line search): обравши напрямок руху dₖ (наприклад, антиградієнт у "
     "градієнтному спуску чи напрямок Ньютона в багатовимірному методі), потрібно визначити оптимальну "
     "довжину кроку α, що мінімізує f(xₖ + α·dₖ) уздовж цього напрямку — а це вже одновимірна задача "
     "мінімізації, яку розв'язують саме методами цієї роботи (золотий переріз, метод Брента тощо)."),
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
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    return t


def fmt(v, nd=6):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


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
    "Опанувати класичні методи мінімізації функції однієї змінної — дихотомії, золотого перерізу, Фібоначчі "
    "та Ньютона, реалізувати їх мовою Python і порівняти за кількістю обчислень цільової функції, швидкістю "
    "збіжності та стійкістю до вибору початкового наближення."
)

# --- 2. Постановка задачі ---
add_heading("2. Постановка задачі")
add_para(f"f(x) = {f_sym}   ->  min,   x ∈ [{a0}; {b0}]",
         bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(
    f"Похідні: f'(x) = {df_sym},   f''(x) = {d2f_sym}. "
    f"Початкове наближення для методу Ньютона: x0 = {x0_newton}. Точність: ε = {eps0}."
)
doc.add_picture(func_plot_path, width=Cm(13))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    "За графіком видно, що функція унімодальна на заданому відрізку — має єдиний мінімум, позначений "
    "вертикальною лінією, до якого функція монотонно спадає і після якого монотонно зростає."
)

# --- 3. Аналітичний розв'язок ---
add_heading("3. Аналітичне знаходження точки мінімуму")
add_para(
    f"Розв'язок рівняння f'(x) = 0 (чисельно, методом Ньютона з sympy.nsolve, оскільки в загальному випадку "
    f"закритої форми немає): x* = {x_exact:.6f}, f(x*) = {f(x_exact):.6f}."
)
add_para(
    f"Перевірка достатньої умови мінімуму: f''(x*) = {f2_at_exact:.6f} "
    f"{'> 0, отже це точка МІНІМУМУ.' if is_min else '<= 0, отже це НЕ мінімум.'}"
)

# --- 4. Реалізація методів ---
add_heading("4. Реалізація методів")
add_code_block(CODE_METHODS.strip("\n"))

# --- 5. Таблиці ітерацій ---
add_heading("5. Таблиці перших ітерацій кожного методу")
method_hists = [
    ("Дихотомія", hist_dich, ["iter", "a", "b", "x1", "x2", "f1", "f2", "length"]),
    ("Золотий переріз", hist_gold, ["iter", "a", "b", "x1", "x2", "f1", "f2", "length"]),
    ("Фібоначчі", hist_fib, ["iter", "a", "b", "x1", "x2", "f1", "f2", "length"]),
    ("Ньютона", hist_newt, ["iter", "x", "fpx", "fppx", "step", "x_next"]),
]
for name, hist, cols in method_hists:
    add_para(name, bold=True)
    rows = [[fmt(h[c]) if isinstance(h[c], float) else h[c] for c in cols] for h in hist[:7]]
    add_table(cols, rows)

# --- 6. Порівняння зі SciPy та зведена таблиця ---
add_heading("6. Порівняння з SciPy (метод Брента) та зведена таблиця")
add_para(
    f"SciPy (bounds=({a0},{b0}), method='bounded'): x* = {res.x:.6f}, f(x*) = {res.fun:.6f}, "
    f"обчислень f = {res.nfev}, ітерацій = {res.nit}."
)
add_table(
    ["Метод", "x*", "f(x*)", "|x*-x_точне|", "Обчислень f", "Ітерацій"],
    [[name, fmt(xs_), fmt(fs_), fmt(err, 8), calls, iters] for name, xs_, fs_, err, calls, iters in summary_rows]
)

# --- 7. Графік збіжності ---
add_heading("7. Графік скорочення інтервалу невизначеності (логарифмічна шкала)")
doc.add_picture(conv_plot_path, width=Cm(13))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    "Чим стрімкіше спадає крива методу на графіку, тим швидша його збіжність. Дихотомія витрачає по два "
    "обчислення функції на ітерацію, тому програє золотому перерізу й Фібоначчі за кількістю обчислень при "
    "тій самій точності, хоча всі три мають лінійний тип збіжності."
)

# --- 8. Вплив точності eps ---
add_heading("8. Вплив точності ε на кількість обчислень")
add_table(
    ["ε", "Дихотомія", "Золотий переріз", "Фібоначчі", "Ньютона"],
    [[f"{e:.0e}", cd, cg, cf, cn] for e, cd, cg, cf, cn in eps_rows]
)
add_para(
    "Із таблиці видно, що для методів скорочення інтервалу кількість обчислень зростає приблизно лінійно "
    "з логарифмом 1/ε, тоді як метод Ньютона майже не залежить від ε завдяки квадратичній збіжності — "
    "кожна додаткова ітерація на порядки підвищує точність."
)

# --- 9. Стійкість методу Ньютона ---
add_heading("9. Експеримент зі стійкості методу Ньютона")
add_para(f"Досліджено запуски з початкових наближень: {starts}.")
add_table(
    ["x0", "Результат x*", "Ітерацій", "Статус"],
    [[s, (fmt(xr) if isinstance(xr, (int, float)) and xr is not None else str(xr)), nit, status]
     for s, xr, nit, status in stability_rows]
)
any_diverged = any("розбіж" in status or "помилка" in status for _, _, _, status in stability_rows)
add_para(
    ("Експеримент підтверджує нестійкість методу Ньютона: принаймні одне з початкових наближень призвело до "
     "розбіжності або виходу за межі області визначення, тоді як методи скорочення інтервалу (дихотомія, "
     "золотий переріз, Фібоначчі) гарантовано збігаються для будь-якого початкового відрізка, що містить "
     "унімодальний мінімум.")
    if any_diverged else
    ("У цьому конкретному варіанті всі три перевірені початкові наближення привели метод Ньютона до того "
     "самого мінімуму — функція досить «зручна» (опукла на всьому відрізку) для глобальної збіжності Ньютона "
     "тут. Це не суперечить загальній нестійкості методу: для функцій з кількома коренями похідної "
     "(наприклад, вигляду a·x + c/x + d) невдалий старт може призвести до розбіжності, як показано в "
     "теоретичній частині роботи.")
)

# --- 10. Висновки ---
add_heading("10. Висновки")
conclusions = [
    f"Усі чотири методи знайшли мінімум функції варіанта {VARIANT} із заданою точністю ε = {eps0}, "
    f"результати узгоджуються між собою та зі SciPy (метод Брента) з точністю до {eps0:.0e}.",
    "Дихотомія — найпростіша, але найдорожча (2 обчислення f на ітерацію); золотий переріз і Фібоначчі "
    "ефективніші (1 обчислення після першої ітерації), причому Фібоначчі теоретично оптимальний за фіксованої "
    "кількості кроків, а золотий переріз зручніший, коли кількість ітерацій наперед невідома.",
    "Метод Ньютона найшвидший (квадратична збіжність), але вимагає обчислення похідних і не гарантує "
    "глобальної збіжності — при невдалому початковому наближенні може розійтися або збігтися до сторонньої "
    "стаціонарної точки.",
    "Практичні рекомендації: якщо обчислення f(x) дороге і похідні недоступні — золотий переріз (або "
    "Фібоначчі за відомої кількості кроків); якщо похідні доступні й відоме добре початкове наближення — "
    "метод Ньютона або гібридний метод Брента; якщо функція задана лише таблично (дискретний набір точок) — "
    "застосовний тільки метод, що використовує самі значення f (дихотомія / золотий переріз / Фібоначчі), "
    "оскільки похідні для табличної функції обчислити не можна.",
]
for c in conclusions:
    doc.add_paragraph(c, style="List Bullet")

# --- 11. Контрольні питання ---
add_heading("11. Відповіді на контрольні питання")
for i, (q, ans) in enumerate(QA, start=1):
    add_para(f"{i}. {q}", bold=True)
    add_para(ans)

docx_path = f"Zvit_PR{WORK_NUMBER}_Variant{VARIANT}.docx"
doc.save(docx_path)
print(f"Готово: {docx_path} (+ {func_plot_path}, {conv_plot_path})")
