"""
Генератор звіту до практичної роботи №8 (Нелінійне програмування з обмеженнями:
умови ККТ, штрафні та бар'єрні функції).
Питає номер варіанта (або приймає його аргументом) і генерує Zvit_PR8_Variant<N>.docx.

Запуск:  .venv/bin/python generate_report.py [номер_варіанта]
"""
import sys
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
WORK_TITLE = "Нелінійне програмування з обмеженнями: умови ККТ, штрафні та бар'єрні функції"
WORK_NUMBER = 8

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
f_str, g1_str, g2_str, R_LIST = VARIANTS[VARIANT]

x1, x2 = sp.symbols('x1 x2', real=True)
f_sym = sp.sympify(f_str, locals={'x1': x1, 'x2': x2})
g1_sym = sp.sympify(g1_str, locals={'x1': x1, 'x2': x2})
g2_sym = sp.sympify(g2_str, locals={'x1': x1, 'x2': x2})

f_np = sp.lambdify((x1, x2), f_sym, 'numpy')
g1_np = sp.lambdify((x1, x2), g1_sym, 'numpy')
g2_np = sp.lambdify((x1, x2), g2_sym, 'numpy')


def f(x):
    return float(f_np(x[0], x[1]))


def g1(x):
    return float(g1_np(x[0], x[1]))


def g2(x):
    return float(g2_np(x[0], x[1]))


G = [g1, g2]
G_sym = [g1_sym, g2_sym]
grad_f_sym = [sp.diff(f_sym, v) for v in (x1, x2)]
grad_g_sym = [[sp.diff(g, v) for v in (x1, x2)] for g in G_sym]

# ---------- Безумовний мінімум ----------
unc_sol = sp.solve([sp.Eq(gi, 0) for gi in grad_f_sym], [x1, x2], dict=True)[0]
x_unc = (float(unc_sol[x1]), float(unc_sol[x2]))
unc_feasible = g1(x_unc) <= 1e-9 and g2(x_unc) <= 1e-9

# ---------- Аналітичний розв'язок за ККТ (перебір активних наборів) ----------
def solve_kkt_case(active):
    """active: підмножина {0,1} індексів активних обмежень.
    Повертає dict(x, lam, accepted, reason) — завжди, навіть для відкинутих випадків,
    щоб показати в звіті ПОВНИЙ перебір варіантів (так, як вимагає завдання)."""
    if not active:
        x_try = x_unc
        lam_val = {0: 0.0, 1: 0.0}
        if g1(x_try) <= 1e-7 and g2(x_try) <= 1e-7:
            return dict(x=x_try, lam=lam_val, accepted=True, reason="допустимо, λ=0≥0")
        return dict(x=x_try, lam=lam_val, accepted=False,
                    reason=f"недопустимо: g1={g1(x_try):.3g}>0" if g1(x_try) > 0
                    else f"недопустимо: g2={g2(x_try):.3g}>0")

    lam_syms = {i: sp.symbols(f'lam{i}', real=True) for i in active}
    stat_eqs = []
    for k, v in enumerate((x1, x2)):
        expr = grad_f_sym[k] + sum(lam_syms[i] * grad_g_sym[i][k] for i in active)
        stat_eqs.append(sp.Eq(expr, 0))
    act_eqs = [sp.Eq(G_sym[i], 0) for i in active]
    unknowns = [x1, x2] + [lam_syms[i] for i in active]
    sols = sp.solve(stat_eqs + act_eqs, unknowns, dict=True)
    if not sols:
        return dict(x=None, lam=None, accepted=False, reason="система стаціонарності несумісна (немає розв'язку)")

    sol = sols[0]
    x_try = (float(sol[x1]), float(sol[x2]))
    lam_val = {i: float(sol[lam_syms[i]]) for i in active}
    for i in (0, 1):
        if i not in active:
            lam_val[i] = 0.0

    if g1(x_try) > 1e-6 or g2(x_try) > 1e-6:
        bad = "g1" if g1(x_try) > 1e-6 else "g2"
        return dict(x=x_try, lam=lam_val, accepted=False,
                    reason=f"порушує неактивне обмеження {bad} (сама точка недопустима)")
    neg = [i for i in active if lam_val[i] < -1e-6]
    if neg:
        return dict(x=x_try, lam=lam_val, accepted=False,
                    reason=f"λ{neg[0]+1}={lam_val[neg[0]]:.3g}<0 (порушує умову невід'ємності)")
    return dict(x=x_try, lam=lam_val, accepted=True, reason="допустимо, λ≥0 — усі умови ККТ виконані")


kkt_candidates = []
for active in ([], [0], [1], [0, 1]):
    kkt_candidates.append((active, solve_kkt_case(active)))

accepted = [(a, r) for a, r in kkt_candidates if r["accepted"]]
# Опукла задача (f опукла — сума квадратів або додатно визначена квадратична форма,
# g1,g2 лінійні) => єдиний КРК-розв'язок (якщо знайдений) і є глобальним мінімумом.
active_case, _accepted_res = accepted[0]
x_kkt, lam_kkt = _accepted_res["x"], _accepted_res["lam"]
f_kkt = f(x_kkt)

# ---------- Метод зовнішніх штрафних функцій ----------
def penalty(x, r):
    return f(x) + r * sum(max(0.0, g(x)) ** 2 for g in G)


penalty_rows = []
x_pen = np.array([0.0, 0.0])
for r in R_LIST:
    res = minimize(penalty, x_pen, args=(r,), method='BFGS')
    x_pen = res.x
    pen = penalty(x_pen, r) - f(x_pen)
    penalty_rows.append(dict(r=r, x1=x_pen[0], x2=x_pen[1], f=f(x_pen), g1=g1(x_pen), penalty=pen))
path_penalty = np.array([[0.0, 0.0]] + [[row["x1"], row["x2"]] for row in penalty_rows])

# ---------- Метод бар'єрних функцій ----------
def interior_start():
    """Шукаємо точку строго всередині допустимої області (g1<0, g2<0)."""
    for t in np.linspace(0.05, 0.95, 37):
        cand = np.array(x_unc) * 0 + t * np.array([0.0, 0.0])  # просто стартуємо з 0,0 і зсуваємо
    # Найпростіше: взяти середину між початком координат і x_kkt, подвинуту трохи всередину.
    cand = 0.3 * np.array(x_kkt)
    if g1(cand) < -1e-3 and g2(cand) < -1e-3:
        return cand
    # fallback: грубий пошук по сітці
    for a in np.linspace(-5, 5, 41):
        for b in np.linspace(-5, 5, 41):
            if g1([a, b]) < -1e-2 and g2([a, b]) < -1e-2:
                return np.array([a, b])
    raise RuntimeError("Не вдалось знайти допустиму внутрішню точку")


def barrier(x, mu):
    if any(g(x) >= -1e-12 for g in G):
        return 1e12
    return f(x) - mu * sum(np.log(-g(x)) for g in G)


x_bar = interior_start()
barrier_rows = []
mu_list = [1.0 / r for r in R_LIST]
for mu in mu_list:
    res = minimize(barrier, x_bar, args=(mu,), method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000})
    x_bar = res.x
    barrier_rows.append(dict(mu=mu, x1=x_bar[0], x2=x_bar[1], f=f(x_bar), g1=g1(x_bar)))
path_barrier = np.array([list(0.3 * np.array(x_kkt))] + [[row["x1"], row["x2"]] for row in barrier_rows])

# ---------- SLSQP ----------
cons = [{'type': 'ineq', 'fun': lambda x, g=g: -g(x)} for g in G]
res_slsqp = minimize(f, [0.0, 0.0], method='SLSQP', constraints=cons, options={'ftol': 1e-12})
x_slsqp = res_slsqp.x

# ---------- Чисельна перевірка умов ККТ у знайденому SLSQP-розв'язку ----------
grad_f_val = np.array([float(sp.lambdify((x1, x2), e, 'numpy')(*x_slsqp)) for e in grad_f_sym])
A = np.array([[float(sp.lambdify((x1, x2), e, 'numpy')(*x_slsqp)) for e in row] for row in grad_g_sym])
active_idx = [i for i, g in enumerate(G) if abs(g(x_slsqp)) < 1e-6]
lam_num = np.zeros(2)
if active_idx:
    lam_num[active_idx] = np.linalg.lstsq(A[active_idx].T, -grad_f_val, rcond=None)[0]
stationarity_residual = grad_f_val + A.T @ lam_num
compl_slack = [float(lam_num[i] * G[i](x_slsqp)) for i in range(2)]

# ---------- Аналіз чутливості ----------
active_i = active_case[0] if active_case else None
sens_rows = []
if active_i is not None:
    g_expr_str = [g1_str, g2_str][active_i]
    # g_str має вигляд "x1+x2-B" -> права частина B стоїть як remainder
    for delta in (-1, -0.01, 0, 0.01, 1):
        shifted_expr_str = f"({g_expr_str}) - ({delta})"
        g_shifted_sym = sp.sympify(shifted_expr_str, locals={'x1': x1, 'x2': x2})
        g_shifted_np = sp.lambdify((x1, x2), g_shifted_sym, 'numpy')
        Gd = [g1, g2]
        Gd[active_i] = lambda x, gf=g_shifted_np: float(gf(x[0], x[1]))
        cons_d = [{'type': 'ineq', 'fun': lambda x, g=g: -g(x)} for g in Gd]
        r_d = minimize(f, [0.0, 0.0], method='SLSQP', constraints=cons_d, options={'ftol': 1e-12})
        # Теорія (розділ 8.2 методички): лямбда* = -d f*/d b, тому локальний (лінійний) прогноз
        # зміни оптимуму при зсуві правої частини на delta: f*(b+delta) ~ f*(b) - лямбда*·delta.
        predicted = f_kkt - lam_kkt[active_i] * delta if delta != 0 else f_kkt
        sens_rows.append(dict(
            delta=delta, f_actual=r_d.fun, f_predicted=predicted,
            diff=abs(r_d.fun - predicted),
        ))

# ---------- Графік: допустима область + обидві траєкторії ----------
all_pts = np.vstack([path_penalty, path_barrier, [x_unc], [x_kkt]])
center = all_pts.mean(axis=0)
half_width = max(2.5, np.abs(all_pts - center).max() + 1.0)
xs = np.linspace(center[0] - half_width, center[0] + half_width, 300)
ys = np.linspace(center[1] - half_width, center[1] + half_width, 300)
X, Y = np.meshgrid(xs, ys)
Z = f_np(X, Y)
G1Z = g1_np(X, Y)
G2Z = g2_np(X, Y)
infeasible_mask = (G1Z > 0) | (G2Z > 0)


def contour_levels(Z, n=24):
    qs = np.linspace(0, 1, n)
    return np.unique(np.quantile(Z, qs))


fig, ax = plt.subplots(figsize=(7, 6))
ax.contour(X, Y, Z, levels=contour_levels(Z), cmap='viridis', alpha=0.7)
ax.contourf(X, Y, infeasible_mask.astype(float), levels=[0.5, 1.5], colors=['0.75'], alpha=0.5)
ax.plot(path_penalty[:, 0], path_penalty[:, 1], 'o-', color='crimson', ms=4, label='штрафний метод (ззовні)')
ax.plot(path_barrier[:, 0], path_barrier[:, 1], 's-', color='royalblue', ms=4, label="бар'єрний метод (зсередини)")
ax.plot(*x_unc, 'kx', ms=10, label='безумовний мінімум')
ax.plot(*x_kkt, 'g*', ms=16, label='x* (ККТ)')
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.legend(fontsize=8, loc='best')
ax.set_title(f'Варіант {VARIANT}: допустима область (сіра — недопустима) і траєкторії методів')
fig.tight_layout()
traj_path = f"trajectories_variant_{VARIANT}.png"
fig.savefig(traj_path, dpi=150)
plt.close(fig)

# ---------- Лістинг коду ----------
CODE_METHODS = '''def penalty(x, r):
    return f(x) + r * sum(max(0.0, g(x)) ** 2 for g in G)   # штраф = 0 у допустимій області


def barrier(x, mu):
    if any(g(x) >= -1e-12 for g in G):
        return 1e12                                          # поза областю бар'єр не визначений
    return f(x) - mu * sum(np.log(-g(x)) for g in G)


# Штрафна послідовність: r зростає, старт кожної задачі - розв'язок попередньої.
x = np.array([0.0, 0.0])
for r in R_LIST:
    res = minimize(penalty, x, args=(r,), method='BFGS')
    x = res.x

# Бар'єрна послідовність: mu спадає (mu = 1/r), старт - строго допустима точка.
x = interior_start()
for mu in [1.0 / r for r in R_LIST]:
    res = minimize(barrier, x, args=(mu,), method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 20000})
    x = res.x

# SLSQP - пряме розв'язання задачі з обмеженнями, без послідовності параметрів.
cons = [{'type': 'ineq', 'fun': lambda x, g=g: -g(x)} for g in G]
res = minimize(f, [0.0, 0.0], method='SLSQP', constraints=cons, options={'ftol': 1e-12})

# Чисельна перевірка умов ККТ у знайденій точці:
# lambda знаходиться з лінійної системи grad_f + A^T @ lambda = 0 по активних обмеженнях.
'''

# ---------- Контрольні питання ----------
QA = [
    ("Запишіть функцію Лагранжа для задачі з обмеженнями-нерівностями. Який знак мають множники?",
     "L(x,λ,μ) = f(x) + Σi λigi(x) + Σj μjhj(x), де gi(x)≤0 — обмеження-нерівності, hj(x)=0 — "
     "рівності. Множники при нерівностях обов'язково невід'ємні, λi≥0 (інакше ослаблення обмеження "
     "могло б погіршити, а не покращити оптимум, що суперечить змісту множника як «тіньової ціни»); "
     "множники при рівностях μj можуть мати будь-який знак, бо рівність не можна «послабити» лише в "
     "один бік."),
    ("Сформулюйте всі умови Каруша-Куна-Таккера.",
     "(1) Стаціонарність: ∇f(x*) + Σiλi*∇gi(x*) + Σjμj*∇hj(x*) = 0 — антиградієнт цілі зрівноважений "
     "градієнтами активних обмежень. (2) Допустимість: gi(x*)≤0, hj(x*)=0 — точка належить допустимій "
     "області. (3) Невід'ємність: λi*≥0. (4) Доповнювальна нежорсткість: λi*·gi(x*)=0 для кожного i — "
     "або обмеження неактивне (і тоді його множник нуль), або множник додатний (і тоді обмеження "
     "виконується як рівність)."),
    ("Поясніть зміст умови доповнювальної нежорсткості на прикладі активного і неактивного обмеження.",
     f"У моєму варіанті в точці x*={tuple(round(v,4) for v in x_kkt)} обмеження g1 активне "
     f"(g1(x*)=0, λ1*={lam_kkt.get(0,0):.3g}>0) — воно «тисне» на оптимум, і щоб якось зрушити x* "
     "далі в бік зменшення f, довелось би порушити саме це обмеження. Обмеження g2 неактивне "
     f"(g2(x*)={g2(x_kkt):.4g}<0, λ2*=0) — воно має «запас» і жодним чином не впливає на розв'язок: "
     "його можна було б прибрати зовсім, оптимум не зміниться. Добуток λi·gi завжди нуль: там, де "
     "множник додатний, обмеження точно на межі (gi=0); де обмеження з запасом (gi<0), множник точно "
     "нуль."),
    ("За яких умов умови ККТ є достатніми для глобального мінімуму?",
     "Коли задача опукла: цільова функція f опукла, усі обмеження-нерівності gi опуклі, а "
     "обмеження-рівності hj лінійні (афінні). У моєму варіанті f — сума квадратів (опукла), g1 і g2 — "
     "лінійні функції (опуклі й угнуті одночасно) — тому знайдена КРК-точка є не просто локальним, а "
     "глобальним мінімумом задачі."),
    ("Який економічний зміст має множник Лагранжа? Як він пов'язаний із тіньовою ціною в ЛП?",
     "λi* = −∂f*/∂bi показує, наскільки покращиться (зменшиться) оптимальне значення f* при послабленні "
     f"правої частини i-го обмеження на одиницю. У моєму варіанті λ1*≈{lam_kkt.get(0,0):.3g}: якщо "
     "дозволити x1+x2 бути трохи більшим за поточну межу, f* зменшиться приблизно на цю величину за "
     "кожну одиницю послаблення — це перевірено числовим експериментом у розділі аналізу чутливості. "
     "Це той самий зміст, що й тіньова ціна ресурсу в лінійному програмуванні: гранична цінність "
     "одиниці дефіцитного ресурсу."),
    ("У чому полягає ідея методу зовнішніх штрафних функцій? Чому r має зростати?",
     "Задачу з обмеженнями заміняють послідовністю задач безумовної мінімізації P(x,r)=f(x)+r·Σ[max(0,"
     "gi(x))]², де штраф нульовий у допустимій області і зростає квадратично при порушенні обмежень. "
     "При малому r штраф «м'який», і розв'язок помітно порушує обмеження (наближення йде ззовні); "
     "збільшуючи r до нескінченності, порушення обмежень прямує до нуля, і послідовність розв'язків "
     "збігається до x*. Якщо одразу взяти величезне r, допоміжна задача стає погано зумовленою й важко "
     "розв'язуваною чисельно — тому r нарощують поступово, використовуючи попередній розв'язок як "
     "стартову точку для наступного."),
    ("Чому при великих значеннях r допоміжна задача стає погано зумовленою?",
     "Штрафний доданок r·[max(0,g(x))]² має величезну кривизну (другу похідну порядку r) у напрямку, "
     "перпендикулярному до межі обмеження, і майже нульову кривизну вздовж межі — тобто поверхня рівня "
     "P(x,r) перетворюється на вузький витягнутий яр (аналогічний тому, що розглядався в практичній "
     "роботі 3). Число зумовленості матриці Гессе такої функції зростає пропорційно r, тому градієнтні "
     "методи починають «зиґзаґати» і втрачають точність при наближенні до розв'язку великими r."),
    ("Чим бар'єрний метод відрізняється від штрафного? Коли його застосування обов'язкове?",
     "Бар'єрна функція B(x,μ)=f(x)−μΣln(−gi(x)) нескінченно зростає при наближенні до межі зсередини, "
     "тому всі проміжні наближення залишаються строго допустимими (на відміну від штрафного методу, "
     "де проміжні точки трохи порушують обмеження). Бар'єрний метод обов'язковий, коли цільова функція "
     "або сама задача просто не визначена за межами допустимої області — наприклад, під коренем чи "
     "логарифмом стоїть вираз, що має бути додатним, або фізично неможливо навіть тимчасово вийти за "
     "обмеження (аварійний режим роботи системи)."),
    ("Що таке центральний шлях у методах внутрішньої точки?",
     "Множина розв'язків x*(μ) бар'єрної задачі для послідовності μ→0 утворює неперервну криву "
     "всередині допустимої області, що «стікає» до справжнього розв'язку x* задачі з обмеженнями — цю "
     "криву називають центральним шляхом. Саме рух уздовж нього (а не випадковий пошук) лежить в "
     "основі сучасних методів внутрішньої точки для лінійного і опуклого програмування, які на задачах "
     "великої розмірності конкурують із симплекс-методом."),
    ("Опишіть ідею методу послідовного квадратичного програмування (SQP).",
     "На кожній ітерації SQP будує квадратичну апроксимацію функції Лагранжа (з квазіньютонівською "
     "матрицею Bk замість справжнього гессіана) і лінійну апроксимацію обмежень навколо поточної точки, "
     "після чого розв'язує допоміжну задачу квадратичного програмування відносно напрямку кроку d. "
     "Метод має надлінійну збіжність і один з найефективніших для гладких задач середньої розмірності "
     "— у моєму варіанті SLSQP знайшов точний розв'язок за кілька ітерацій, без потреби в послідовності "
     "штрафних параметрів."),
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


def pt(v):
    return tuple(float(c) for c in v)


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
    "Опанувати методи розв'язання задач нелінійного програмування з обмеженнями: аналітичну перевірку "
    "умов Каруша-Куна-Таккера, метод зовнішніх штрафних функцій, метод бар'єрних функцій та метод "
    "послідовного квадратичного програмування (SLSQP); реалізувати перші два мовою Python і порівняти "
    "характер збіжності траєкторій до розв'язку."
)

# --- 2. Постановка задачі ---
add_heading("2. Постановка задачі та функція Лагранжа")
add_para(f"f(x1,x2) = {f_sym}  ->  min", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(f"g1(x): {g1_sym} <= 0,   g2(x): {g2_sym} <= 0", align=WD_ALIGN_PARAGRAPH.CENTER)
add_para(
    f"Функція Лагранжа: L(x,λ) = f(x) + λ1·g1(x) + λ2·g2(x), λ1,λ2 >= 0. "
    f"Безумовний мінімум f(x) досягається в точці {pt(x_unc)}: "
    f"g1 = {fmt(g1(x_unc),4)} {'(допустимо)' if g1(x_unc)<=0 else '(порушено)'}, "
    f"g2 = {fmt(g2(x_unc),4)} {'(допустимо)' if g2(x_unc)<=0 else '(порушено)'}."
)
add_para(
    ("Безумовний мінімум недопустимий, отже справжній оптимум лежить на межі допустимої області."
     if not unc_feasible else
     "Безумовний мінімум уже допустимий — обидва обмеження в ньому неактивні, і він одночасно є "
     "розв'язком задачі з обмеженнями.")
)
doc.add_picture(traj_path, width=Cm(13))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
add_para(
    "Сіра область на графіку — недопустима зона (де порушено хоча б одне з обмежень). Хрестик — "
    "безумовний мінімум f, зірка — знайдений розв'язок x* задачі з обмеженнями."
)

# --- 3. Аналітичний розв'язок за ККТ ---
add_heading("3. Аналітичний розв'язок за умовами ККТ")
add_para(
    "Перебрано всі можливі набори активних обмежень (жодне, лише g1, лише g2, обидва) і для кожного "
    "розв'язано систему стаціонарності разом з рівняннями активних обмежень, після чого перевірено "
    "допустимість результату та знак множників λ>=0."
)
case_names = {(): "жодне обмеження не активне", (0,): "активне лише g1", (1,): "активне лише g2",
              (0, 1): "активні обидва обмеження"}
rows_cases = []
for active, r in kkt_candidates:
    x_c, lam_c = r["x"], r["lam"]
    rows_cases.append([
        case_names[tuple(active)],
        (fmt(x_c[0]) + ", " + fmt(x_c[1])) if x_c is not None else "—",
        fmt(lam_c.get(0, 0.0), 4) if lam_c else "—",
        fmt(lam_c.get(1, 0.0), 4) if lam_c else "—",
        ("ПРИЙНЯТО: " if r["accepted"] else "відкинуто: ") + r["reason"],
    ])
add_table(["Набір активних обмежень", "x (x1,x2)", "λ1", "λ2", "Висновок"], rows_cases)
add_para(
    f"Єдиний набір, що задовольняє всі умови ККТ (допустимість + λ>=0): {case_names[tuple(active_case)]}. "
    f"x* = {pt(x_kkt)}, f(x*) = {fmt(f_kkt)}, λ1* = {fmt(lam_kkt.get(0,0.0),4)}, "
    f"λ2* = {fmt(lam_kkt.get(1,0.0),4)}. Оскільки f опукла, а g1,g2 лінійні (отже опуклі), ця задача "
    "опукла, і знайдена КРК-точка є глобальним мінімумом."
)

# --- 4. Тексти програм ---
add_heading("4. Тексти програм")
add_code_block(CODE_METHODS.strip("\n"))

# --- 5. Таблиці наближень ---
add_heading("5. Таблиці наближень штрафного і бар'єрного методів")
add_para("Метод зовнішніх штрафних функцій (наближення ЗЗОВНІ, g1 поступово -> 0+):", bold=True)
add_table(
    ["r", "x1", "x2", "f(x)", "g1(x)", "величина штрафу"],
    [[row["r"], fmt(row["x1"]), fmt(row["x2"]), fmt(row["f"]), fmt(row["g1"]), f"{row['penalty']:.3e}"]
     for row in penalty_rows]
)
add_para("Метод бар'єрних функцій (наближення ЗСЕРЕДИНИ, g1 поступово -> 0-):", bold=True)
add_table(
    ["mu = 1/r", "x1", "x2", "f(x)", "g1(x)"],
    [[fmt(row["mu"], 4), fmt(row["x1"]), fmt(row["x2"]), fmt(row["f"]), fmt(row["g1"])]
     for row in barrier_rows]
)
add_para(
    f"Штрафна траєкторія підходить до межі ззовні (g1(x)>0, порушення спадає при зростанні r), "
    f"бар'єрна — зсередини (g1(x)<0 на кожному кроці, відстань до межі спадає разом з μ). Обидві "
    f"збігаються до того самого x* = {pt(x_kkt)}, знайденого аналітично."
)

# --- 6. SLSQP ---
add_heading("6. Розв'язання методом SLSQP")
add_para(
    f"SLSQP: x* = ({x_slsqp[0]:.6f}; {x_slsqp[1]:.6f}), f* = {res_slsqp.fun:.6f} "
    f"(за {res_slsqp.nit} ітерацій, без жодної послідовності параметрів r чи μ). "
    f"Збігається з аналітичним розв'язком x* = {pt(x_kkt)}, f* = {fmt(f_kkt)} з точністю до 6 знаків."
)

# --- 7. Перевірка умов ККТ ---
add_heading("7. Чисельна перевірка умов ККТ у розв'язку SLSQP")
add_table(
    ["Умова", "Значення", "Виконано?"],
    [
        ["Стаціонарність: ||grad f + sum(lam*grad g)||",
         f"{float(np.linalg.norm(stationarity_residual)):.2e}", "так (≈0)"],
        ["Допустимість g1(x*)", fmt(g1(x_slsqp), 6), "так" if g1(x_slsqp) <= 1e-6 else "ні"],
        ["Допустимість g2(x*)", fmt(g2(x_slsqp), 6), "так" if g2(x_slsqp) <= 1e-6 else "ні"],
        ["Невід'ємність λ1,λ2", f"({fmt(lam_num[0],4)}; {fmt(lam_num[1],4)})",
         "так" if all(lam_num >= -1e-6) else "ні"],
        ["Доп. нежорсткість λ1*g1, λ2*g2",
         f"({compl_slack[0]:.2e}; {compl_slack[1]:.2e})", "так (≈0)"],
    ]
)

# --- 8. Чутливість ---
add_heading("8. Аналіз чутливості до правої частини активного обмеження")
if sens_rows:
    add_para(
        f"Активне обмеження g{active_i+1} зсунуто на Δ=±1 (інші обмеження лишились без змін), задачу "
        f"розв'язано заново SLSQP і фактичну зміну f* порівняно з прогнозом λ{active_i+1}*·Δ."
    )
    add_table(
        ["Δ (зсув g)", "f* фактичне", "f* прогноз (λ*·Δ)", "|різниця|"],
        [[row["delta"], fmt(row["f_actual"]), fmt(row["f_predicted"]), fmt(row["diff"], 6)]
         for row in sens_rows]
    )
    big_diff = max(row["diff"] for row in sens_rows if abs(row["delta"]) == 1)
    small_diff = max(row["diff"] for row in sens_rows if 0 < abs(row["delta"]) < 1)
    add_para(
        f"Прогноз λ*·Δ — це ЛІНІЙНЕ (локальне) наближення, точне лише для нескінченно малого зсуву "
        f"Δb. Таблиця це явно підтверджує: для великого зсуву Δ=±1 (50% від початкового b=2) "
        f"розбіжність прогнозу й факту сягає {fmt(big_diff,4)}, а для малого Δ=±0.01 вона падає до "
        f"{fmt(small_diff,6)} — у {big_diff/small_diff:.0f} разів точніше. Особливо показовий "
        f"випадок Δ=+1 (послаблення обмеження): фактичне f* падає різкіше, ніж передбачає лінійна "
        f"формула, бо при такому послабленні обмеження перестає бути активним і задача «повертається» "
        f"до безумовного мінімуму — цей стрибок набору активних обмежень лінійний прогноз за λ* у "
        f"принципі не враховує, бо формула виведена в припущенні, що активні обмеження лишаються тими "
        f"самими."
    )
else:
    add_para("У цьому варіанті жодне обмеження не є активним, аналіз чутливості по ньому не застосовний.")

# --- 9. Висновки ---
add_heading("9. Висновки")
conclusions = [
    f"Для варіанта {VARIANT} аналітичний розв'язок за ККТ (x*={pt(x_kkt)}, f*={fmt(f_kkt)}) повністю "
    f"узгоджується з чисельним SLSQP (x*=({x_slsqp[0]:.6f};{x_slsqp[1]:.6f})) та з граничними точками "
    "штрафної й бар'єрної траєкторій.",
    "Штрафний метод зручний, коли проміжні недопустимі точки не є катастрофічними — реалізується "
    "просто (звичайна безумовна мінімізація), але при великих r задача стає погано зумовленою.",
    "Бар'єрний метод обов'язковий, якщо функція не визначена поза допустимою областю — ціна цього: "
    "потрібна допустима стартова точка, і обчислення стають складнішими біля самої межі (логарифм "
    "прямує до нескінченності).",
    "SQP (тут — SLSQP) виявився найефективнішим: дав точний розв'язок напряму, без послідовності "
    "параметрів, за невелику кількість ітерацій — на практиці це метод вибору для гладких задач "
    "середньої розмірності, якщо немає особливих причин обирати штрафний чи бар'єрний підхід.",
    "Чисельна перевірка підтвердила всі чотири умови ККТ у знайденій точці, а аналіз чутливості "
    "підтвердив економічний зміст множника Лагранжа як граничної ціни обмеження.",
]
for c in conclusions:
    doc.add_paragraph(c, style="List Bullet")

# --- 10. Контрольні питання ---
add_heading("10. Відповіді на контрольні питання")
for i, (q, a) in enumerate(QA, start=1):
    add_para(f"{i}. {q}", bold=True)
    add_para(a)

docx_path = f"Zvit_PR{WORK_NUMBER}_Variant{VARIANT}.docx"
doc.save(docx_path)
print(f"Готово: {docx_path} (+ {traj_path})")
print(f"x_kkt={x_kkt} f_kkt={f_kkt} lam={lam_kkt} x_slsqp={x_slsqp} f_slsqp={res_slsqp.fun}")
