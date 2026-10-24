import sys
from pathlib import Path
import numpy as np, pandas as pd

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
rng = np.random.default_rng(SEED)
N = 30_000
sig = lambda x: 1 / (1 + np.exp(-x))

# ---------------- perfil del cliente ----------------
DEPTOS = ["Santa Cruz", "La Paz", "Cochabamba", "Beni", "Tarija", "Chuquisaca", "Oruro", "Potosí", "Pando"]
P_DEP = [0.46, 0.17, 0.14, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01]
dep = rng.choice(len(DEPTOS), N, p=P_DEP)
sex = rng.choice([1, 2], N, p=[0.47, 0.53])
age = np.clip(np.round(21 + rng.gamma(4.2, 4.3, N)), 21, 75).astype(int)
edu = rng.choice([1, 2, 3, 4], N, p=[0.17, 0.50, 0.29, 0.04])
mar = np.where(age < 27, rng.choice([1, 2, 3], N, p=[0.22, 0.75, 0.03]), rng.choice([1, 2, 3], N, p=[0.58, 0.37, 0.05]))
EMP = {1: "Dependiente", 2: "Independiente", 3: "Comerciante", 4: "Jubilado"}
emp = np.where(age >= 60, rng.choice([1, 2, 3, 4], N, p=[0.25, 0.15, 0.20, 0.40]), rng.choice([1, 2, 3, 4], N, p=[0.55, 0.22, 0.23, 0.0]))

# riesgo latente (no observado) y propensión a pagar el total
risk = rng.normal(0, 1, N) + 0.12 * (edu == 3) - 0.15 * (edu == 1) + 0.20 * (emp == 3) + 0.10 * (emp == 2) \
       - 0.10 * (emp == 1) - 0.006 * (age - 38) + 0.08 * np.isin(dep, [3, 8])  # Beni, Pando algo más riesgosos
totalero = sig(-0.4 - 0.9 * risk + 0.4 * (edu == 1))

# límite de crédito (Bs), redondeado a 500
lim = np.exp(np.log(14_000) + 0.75 * rng.normal(0, 1, N) + 0.45 * (edu == 1) - 0.25 * (edu == 3)
             + 0.25 * (emp == 1) + 0.012 * np.clip(age - 25, 0, 30) - 0.18 * risk)
lim = np.clip(np.round(lim / 500) * 500, 2_000, 280_000).astype(int)

# ---------------- 6 meses de historial (m=0 marzo ... m=5 agosto) ----------------
M = 6
status = np.zeros((N, M), int); bill = np.zeros((N, M)); pay = np.zeros((N, M))
u_base = np.clip(rng.beta(2, 2.6, N) + 0.18 * risk, 0.02, 1.05)
s = np.where(rng.random(N) < sig(-2.4 + 1.0 * risk), 1, 0)  # estado inicial (febrero)
s = np.where((s == 0) & (rng.random(N) < totalero), -1, s)
s = np.where((s <= 0) & (rng.random(N) < 0.07), -2, s)
prev_bill = lim * u_base * rng.uniform(0.7, 1.1, N)
shock = rng.normal(0, 1, N)
for m in range(M):
    r_m = risk + 0.35 * shock * (m / M) + rng.normal(0, 0.45, N)  # deterioro gradual de algunos clientes
    new = s.copy()
    aldia = s <= 0
    caer = aldia & (rng.random(N) < sig(-2.55 + 1.1 * r_m + 1.0 * (u_base > 0.85)))
    new[caer] = 1
    ok = aldia & ~caer
    tot = rng.random(N) < totalero
    new[ok & tot] = -1; new[ok & ~tot] = 0
    new[ok & (rng.random(N) < 0.05)] = -2
    mora = s >= 1
    cura = mora & (rng.random(N) < sig(1.35 - 0.85 * r_m - 0.75 * (s - 1)))
    queda = mora & ~cura & (rng.random(N) < 0.30)  # pago parcial: no avanza
    new[cura] = 0; new[mora & ~cura & ~queda] = np.minimum(s[mora & ~cura & ~queda] + 1, 8)
    new[queda] = s[queda]
    # pagos del mes m (sobre la factura anterior)
    p = np.zeros(N)
    p[new == -1] = prev_bill[new == -1]
    mn = new == 0
    p[mn] = prev_bill[mn] * np.where(rng.random(mn.sum()) < 0.35, rng.uniform(0.15, 0.6, mn.sum()), rng.uniform(0.04, 0.12, mn.sum()))
    p[cura] = prev_bill[cura] * rng.uniform(0.10, 0.45, cura.sum()) + lim[cura] * 0.02
    part = queda | ((new >= 1) & (rng.random(N) < 0.30))
    p[part & (new >= 1)] = prev_bill[part & (new >= 1)] * rng.uniform(0.01, 0.08, (part & (new >= 1)).sum())
    p = np.maximum(p, 0)
    # factura del mes m
    util = np.clip(u_base + rng.normal(0, 0.10, N), 0.0, 1.15)
    b = lim * util
    b[new >= 1] = np.maximum(prev_bill[new >= 1] - p[new >= 1], 0) * 1.035 + 45  # interés + cargo por mora (Bs)
    b[new == -2] = rng.choice([0.0, -1.0], (new == -2).sum(), p=[0.8, 0.2]) * rng.uniform(0, 300, (new == -2).sum())
    status[:, m] = new; bill[:, m] = np.round(b); pay[:, m] = np.round(p)
    prev_bill = np.maximum(b, 0); s = new

# ---------------- variable objetivo: incumple en septiembre 2026 ----------------
pay0 = status[:, 5]; nme = (status > 0).sum(1)
ratio = np.where(bill[:, 4] > 0, pay[:, 5] / np.maximum(bill[:, 4], 1), 1).clip(0, 2)
util1 = (bill[:, 5] / lim).clip(-1, 3)
logit = (-2.75 + 0.95 * risk + 0.62 * np.clip(pay0, 0, 3) + 0.20 * nme + 0.55 * util1 - 1.0 * ratio
         + 0.15 * (emp == 3) - 0.10 * (emp == 1) + 0.10 * (pay0 == -2) + rng.normal(0, 0.55, N))
default = (rng.random(N) < sig(logit)).astype(int)

df = pd.DataFrame({"COD_CLIENTE": np.arange(1, N + 1), "MTO_LIMITE_CREDITO": lim, "COD_SEXO": sex, "COD_NIVEL_EDUCATIVO": edu, "COD_ESTADO_CIVIL": mar, "EDAD": age,
                   "DEPARTAMENTO": np.array(DEPTOS)[dep], "TIPO_EMPLEO": pd.Series(emp).map(EMP).values})
for k, m in zip(["EST_PAGO_M0", "EST_PAGO_M1", "EST_PAGO_M2", "EST_PAGO_M3", "EST_PAGO_M4", "EST_PAGO_M5"], range(5, -1, -1)): df[k] = status[:, m]
for i, m in zip(range(1, 7), range(5, -1, -1)): df[f"MTO_FACTURADO_M{i-1}"] = bill[:, m].astype(int)
for i, m in zip(range(1, 7), range(5, -1, -1)): df[f"MTO_PAGADO_M{i-1}"] = pay[:, m].astype(int)
df["IND_INCUMPLE_MES_SIG"] = default

# ---------------- problemas de calidad habituales (para practicar la limpieza) ----------------
idx = rng.choice(N, 300, replace=False); df.loc[idx[:180], "COD_NIVEL_EDUCATIVO"] = rng.choice([0, 5, 6], 180); df.loc[idx[180:], "COD_ESTADO_CIVIL"] = 0
dup = df.sample(25, random_state=SEED).copy(); dup["COD_CLIENTE"] = np.arange(N + 1, N + 26)
df = pd.concat([df, dup], ignore_index=True)

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "dataset_bank_bg.csv"
    out.parent.mkdir(exist_ok=True); df.to_csv(out, index=False); print("guardado:", out)
    print(df.shape, "default:", round(df["IND_INCUMPLE_MES_SIG"].mean(), 4))
    m_ = df[df.EST_PAGO_M0 >= 1]; print("en mora:", len(m_), round(len(m_) / len(df), 4), "paga:", round(1 - m_["IND_INCUMPLE_MES_SIG"].mean(), 4))
    print(df.EST_PAGO_M0.value_counts().sort_index().to_dict())
    print(m_.groupby(m_.EST_PAGO_M0.clip(upper=4))["IND_INCUMPLE_MES_SIG"].agg(["size", "mean"]).round(3))
    print(df[["MTO_LIMITE_CREDITO", "EDAD", "MTO_FACTURADO_M0", "MTO_PAGADO_M0"]].describe().round(0).T)
