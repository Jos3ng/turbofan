"""
================================================================
  MANUTENÇÃO PREDITIVA DE MOTORES TURBOFAN
  Predição de falha com dataset NASA C-MAPSS FD001

  Inspirado no sistema AHEAD da Embraer Serviços & Suporte
  (Aircraft Health Analysis and Diagnosis)

  Autor : José Rubens Mariano Penalva — UFU
  Stack : Python · scikit-learn · pandas · matplotlib
  Data  : NASA C-MAPSS FD001 (Turbofan Engine Degradation)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve,
)

# ── Estilo visual ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0A0E1A",
    "axes.facecolor": "#0F1629",
    "axes.edgecolor": "#1E3A5F",
    "axes.labelcolor": "#A0B8D0",
    "xtick.color": "#A0B8D0",
    "ytick.color": "#A0B8D0",
    "text.color": "#E0EAF4",
    "grid.color": "#1E3A5F",
    "grid.linestyle": "--",
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#0077FF"
TEAL = "#00D4A0"
RED = "#FF4560"
PURPLE = "#A855F7"
AMBER = "#F59E0B"

# ════════════════════════════════════════════════════════════
# 1. CARREGANDO OS DADOS E CONFIGURAÇÕES
# ════════════════════════════════════════════════════════════

print("=" * 60)
print("  NASA C-MAPSS FD001 — Turbofan Engine Degradation")
print("=" * 60)

DATA_FILE = "train_FD001.csv"
train_df = pd.read_csv(DATA_FILE)

print(
    f"\nDataset carregado: "
    f"{train_df['unit'].nunique()} motores | "
    f"{len(train_df):,} ciclos"
)

# Variáveis operacionais
OPERATING = ["op1", "op2", "op3"]

# Filtro automático de sensores constantes
sensor_candidates = [
    'T24', 'T30', 'T50', 'P30', 'Nf', 'Nc',
    'Ps30', 'phi', 'NRf', 'NRc',
    'BPR', 'htBleed', 'W31', 'W32'
]

SENSORS = [
    col for col in sensor_candidates
    if train_df[col].std() > 1e-6
]

SENSOR_LABELS = {
    'T24': 'T24 — Temp. LPC outlet (°R)',
    'T30': 'T30 — Temp. HPC outlet (°R)',
    'T50': 'T50 — Temp. LPT outlet (°R)',
    'P30': 'P30 — Pressão HPC (psia)',
    'Nf': 'Nf  — Velocidade do fan (rpm)',
    'Nc': 'Nc  — Velocidade do núcleo (rpm)',
    'Ps30': 'Ps30 — Pressão estática HPC (psia)',
    'phi': 'phi — Razão combustível/pressão',
    'NRf': 'NRf — Vel. fan corrigida (rpm)',
    'NRc': 'NRc — Vel. núcleo corrigida (rpm)',
    'BPR': 'BPR — Bypass ratio',
    'htBleed': 'htBleed — Sangria de entalpia',
    'W31': 'W31 — Sangria HPT coolant (lbm/s)',
    'W32': 'W32 — Sangria LPT coolant (lbm/s)',
}

print(f"Sensores utilizados ({len(SENSORS)}): {', '.join(SENSORS)}")

# ════════════════════════════════════════════════════════════
# 2. EDA — DEGRADAÇÃO DOS SENSORES AO LONGO DOS CICLOS
# ════════════════════════════════════════════════════════════

train_df['cycle_norm'] = train_df.groupby('unit')['cycle'].transform(
    lambda x: x / x.max()
)

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle(
    "NASA C-MAPSS FD001 — Degradação dos Sensores do Motor Turbofan\n"
    "(Eixo X normalizado pelo ciclo de vida de cada motor)",
    fontsize=12, fontweight="bold", color="#E0EAF4", y=1.01
)

# Seleção dinâmica para o plot caso algum sensor tenha sido descartado
KEY_SENSORS = [s for s in ['T30', 'T50', 'P30', 'phi', 'Nf', 'htBleed'] if s in SENSORS][:6]
COLORS_SENSORS = [BLUE, RED, TEAL, AMBER, PURPLE, "#EC4899"]

for ax, sensor, color in zip(axes.flat, KEY_SENSORS, COLORS_SENSORS):
    sampled_units = train_df['unit'].unique()[:20]
    for uid in sampled_units:
        eng = train_df[train_df['unit'] == uid]
        ax.plot(eng['cycle_norm'], eng[sensor],
                color=color, alpha=0.25, linewidth=0.7)

    mean_traj = train_df.groupby(
        pd.cut(train_df['cycle_norm'], bins=50)
    )[sensor].mean()
    midpoints = [iv.mid for iv in mean_traj.index]
    ax.plot(midpoints, mean_traj.values, color="white",
            linewidth=2, label="Média", zorder=5)

    ax.set_title(SENSOR_LABELS.get(sensor, sensor), fontsize=9, color="#E0EAF4", pad=6)
    ax.set_xlabel("Ciclo de vida normalizado", fontsize=8)
    ax.set_ylabel(sensor, fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("01_sensor_degradation.png", dpi=150, bbox_inches="tight", facecolor="#0A0E1A")
plt.close()
print("\n>> Salvo: 01_sensor_degradation.png")

# ════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING AVANÇADA
# ════════════════════════════════════════════════════════════

WINDOW = 15
RUL_THRESHOLD = 40  # Motores com RUL < 40 ciclos = "Prestes a falhar"


def build_features(df):
    """Adiciona estatísticas de janela deslizante e tendência instantânea."""
    df = df.copy()
    df = df.sort_values(["unit", "cycle"])

    for sensor in SENSORS:
        grp = df.groupby("unit")[sensor]

        # Média Móvel (Valor Atual / Histórico Recente)
        df[f"{sensor}_rmean"] = grp.transform(
            lambda x: x.rolling(WINDOW, min_periods=1).mean()
        )
        # Desvio Padrão Móvel (Variabilidade)
        df[f"{sensor}_rstd"] = grp.transform(
            lambda x: x.rolling(WINDOW, min_periods=1).std()
        ).fillna(0)
        # Delta Instantâneo (Tendência de degradação / Velocidade)
        df[f"{sensor}_delta"] = grp.diff().fillna(0)

    return df


print("\nExecutando Engenharia de Features de alta fidelidade...")
train_feat = build_features(train_df)

# Estrutura final de colunas do modelo
FEATURE_COLS = (
        OPERATING +
        SENSORS +
        [f"{s}_rmean" for s in SENSORS] +
        [f"{s}_rstd" for s in SENSORS] +
        [f"{s}_delta" for s in SENSORS]
)

# Target binário focado em manutenção preventiva real de curto/médio prazo
train_feat['label'] = (train_feat['RUL'] < RUL_THRESHOLD).astype(int)

X = train_feat[FEATURE_COLS].values
y = train_feat['label'].values

print(f"Features totais construídas: {len(FEATURE_COLS)}")
print(f"  → Saudável               : {(y == 0).sum():,} ({(y == 0).mean():.1%})")
print(f"  → Alerta de Falha IMINENTE: {(y == 1).sum():,} ({(y == 1).mean():.1%})")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ════════════════════════════════════════════════════════════
# 4. TREINO E VALIDAÇÃO ROBUSTA DO MODELO
# ════════════════════════════════════════════════════════════

print("\nTreinando Classificador Random Forest Robusto...")
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=5,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("Calculando Cross-Validation Estratificada (5-Fold)...")
cv = cross_val_score(
    model, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1
)

# Salvar importância das features de forma organizada
importance_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance_df.to_csv("feature_importance.csv", index=False)
print(">> Salvo: feature_importance.csv")

# ════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÃO DOS RESULTADOS
# ════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# 5a. Matriz de Confusão
ax0 = fig.add_subplot(gs[0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax0,
            xticklabels=['Saudável', 'Prestes\na falhar'],
            yticklabels=['Saudável', 'Prestes\na falhar'],
            linewidths=0.5, linecolor='#1E3A5F',
            annot_kws={"size": 14, "weight": "bold"})
ax0.set_title("Matriz de Confusão", fontweight="bold", color="#E0EAF4", pad=12)
ax0.set_ylabel("Real", color="#A0B8D0")
ax0.set_xlabel("Previsto", color="#A0B8D0")

# 5b. Top-10 Feature Importance Customizada
ax1 = fig.add_subplot(gs[1])
top10 = importance_df.head(10)

bar_colors = []
for f in top10['feature']:
    if 'rstd' in f:
        bar_colors.append(PURPLE)
    elif 'rmean' in f:
        bar_colors.append(TEAL)
    elif 'delta' in f:
        bar_colors.append(AMBER)
    else:
        bar_colors.append(BLUE)

bars = ax1.barh(range(len(top10)), top10['importance'], color=bar_colors, height=0.6)
ax1.set_yticks(range(len(top10)))
ax1.set_yticklabels(top10['feature'], fontsize=8)
ax1.invert_yaxis()  # Maior importância no topo

for bar, val in zip(bars, top10['importance']):
    ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', color='#E0EAF4', fontsize=8)

ax1.set_title("Top-10 Features Mais Importantes", fontweight="bold", color="#E0EAF4", pad=12)
ax1.set_xlabel("Importância Relativa", color="#A0B8D0")
ax1.grid(True, axis='x', alpha=0.3)

legend_patches = [
    Patch(color=BLUE, label='Sensor Raw'),
    Patch(color=TEAL, label='Rolling Mean'),
    Patch(color=PURPLE, label='Rolling Std'),
    Patch(color=AMBER, label='Delta (Velocidade)')
]
ax1.legend(handles=legend_patches, fontsize=7, loc='lower right')

# 5c. Curva ROC
ax2 = fig.add_subplot(gs[2])
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax2.plot(fpr, tpr, color=BLUE, lw=2.5, label=f'AUC = {auc:.4f}')
ax2.plot([0, 1], [0, 1], '--', color='#4A4A6A', lw=1)
ax2.fill_between(fpr, tpr, alpha=0.12, color=BLUE)
ax2.set_title("Curva ROC", fontweight="bold", color="#E0EAF4", pad=12)
ax2.set_xlabel("Taxa de Falso Positivo", color="#A0B8D0")
ax2.set_ylabel("Taxa de Verdadeiro Positivo", color="#A0B8D0")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

fig.suptitle(
    f"Random Forest Performance | Acurácia: {acc:.2%} · ROC-AUC: {auc:.4f} · CV: {cv.mean():.4f} ± {cv.std():.4f}",
    fontsize=11, fontweight="bold", color="#E0EAF4", y=1.03
)
plt.savefig("02_model_results.png", dpi=150, bbox_inches="tight", facecolor="#0A0E1A")
plt.close()
print(">> Salvo: 02_model_results.png")

# ════════════════════════════════════════════════════════════
# 6. MONITORAMENTO TEMPO REAL — MOTOR ALEATÓRIO
# ════════════════════════════════════════════════════════════

print("\nGerando visualização de telemetria preditiva com amostragem dinâmica...")

# Seleção puramente randômica baseada na frota
sample_unit = train_df["unit"].sample(1, random_state=42).iloc[0]

eng_data = train_feat[train_feat['unit'] == sample_unit].copy()
eng_data['pred_prob'] = model.predict_proba(eng_data[FEATURE_COLS])[:, 1]
eng_data['alert'] = eng_data['pred_prob'] > 0.70  # Threshold crítico de severidade

alert_cycle = eng_data[eng_data['alert']]['cycle'].min() if eng_data['alert'].any() else None

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle(
    f"Motor #{sample_unit} — Linha de Telemetria e Alertas de Saúde\n"
    "(Mapeamento preditivo contínuo baseado no Embraer AHEAD)",
    fontsize=12, fontweight="bold", color="#E0EAF4"
)

# Monitoramento visual dos sensores mais pesados de temperatura (se disponíveis)
if 'T30' in SENSORS: ax1.plot(eng_data['cycle'], eng_data['T30'], color=BLUE, lw=1.8,
                              label='T30 — Temp. HPC outlet (°R)')
if 'T50' in SENSORS: ax1.plot(eng_data['cycle'], eng_data['T50'], color=RED, lw=1.8,
                              label='T50 — Temp. LPT outlet (°R)', alpha=0.85)

if alert_cycle:
    ax1.axvline(alert_cycle, color=AMBER, lw=2, ls='--', label=f'⚠ Alerta Crítico Emitido (Ciclo {alert_cycle})')
    ax1.axvspan(alert_cycle, eng_data['cycle'].max(), color=RED, alpha=0.08)

ax1.set_ylabel("Gradiente Térmico (°R)", color="#A0B8D0")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)

# Gráfico de Risco probabilístico com três níveis intuitivos
ax2.fill_between(eng_data['cycle'], eng_data['pred_prob'], color=RED, alpha=0.3, label='P(Falha Estrutural)')
ax2.plot(eng_data['cycle'], eng_data['pred_prob'], color=RED, lw=1.5)
ax2.axhline(0.3, color=TEAL, lw=1, ls=':', label='Threshold Atenção (0.3)')
ax2.axhline(0.7, color=AMBER, lw=1.2, ls='--', label='Threshold Crítico (0.7)')

if alert_cycle:
    ax2.axvline(alert_cycle, color=AMBER, lw=2, ls='--')

ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Ciclos de Operação Efetivos", color="#A0B8D0")
ax2.set_ylabel("Probabilidade de Falha", color="#A0B8D0")
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.2)

if alert_cycle:
    rul_at_alert = eng_data[eng_data['cycle'] == alert_cycle]['RUL'].values[0]
    max_cycle = eng_data['cycle'].max()
    ax1.text(0.02, 0.97,
             f"ALERTA GATILHADO NO CICLO {alert_cycle} | "
             f"RUL Estimado Real: {rul_at_alert} ciclos restantes | "
             f"Ciclos totais pré-falha: {max_cycle}",
             transform=ax1.transAxes, fontsize=9, color=AMBER, va='top',
             bbox=dict(boxstyle='round,pad=0.4', fc='#0F1629', ec=AMBER, alpha=0.9))

plt.tight_layout()
plt.savefig("03_engine_alert.png", dpi=150, bbox_inches="tight", facecolor="#0A0E1A")
plt.close()
print(">> Salvo: 03_engine_alert.png")


# ════════════════════════════════════════════════════════════
# 7. DIAGNÓSTICO DE MOTOR EM TEMPO REAL (INTERFACIÁVEL)
# ════════════════════════════════════════════════════════════

def diagnosticar_motor(leituras: dict) -> dict:
    """
    Simula o recebimento de uma mensagem estruturada via barramento ARINC/Meteo
    e calcula instantaneamente o status do motor.
    """
    df = pd.DataFrame([leituras])

    # Preenchimento prototipado seguro das colunas operacionais e sensores ausentes
    for op in OPERATING:
        if op not in df.columns: df[op] = 0.0
    for s in SENSORS:
        if s not in df.columns: df[s] = train_df[s].mean()  # Preenche com a média global histórica

        # Como o diagnóstico pontual não possui histórico temporal instantâneo na função pura:
        df[f'{s}_rmean'] = df[s]
        df[f'{s}_rstd'] = 0.0
        df[f'{s}_delta'] = 0.0

    prob = model.predict_proba(df[FEATURE_COLS])[0][1]

    # Classificação em 3 Níveis Realistas de Manutenção Aeronáutica
    if prob < 0.3:
        label = "SAUDÁVEL"
        icone = "🟢"
    elif prob < 0.7:
        label = "ATENÇÃO"
        icone = "🟡"
    else:
        label = "PRESTES A FALHAR"
        icone = "🔴"

    return {"status": label, "risco": f"{prob:.1%}", "icone": icone}


print("\n" + "=" * 60)
print("  DEMO — INTERFACE DE DIAGNÓSTICO (AHEAD EMBRAER SIMULATOR)")
print("=" * 60)

casos = [
    ("Motor Alpha — Assinatura Operacional Estável",
     {'T30': 1592.0, 'T50': 1405.0, 'P30': 551.0, 'phi': 524.0,
      'Nf': 2386.0, 'Nc': 9043.0, 'Ps30': 47.3, 'NRf': 2386.0,
      'NRc': 8135.0, 'BPR': 8.42, 'htBleed': 394.0,
      'W31': 38.9, 'W32': 23.5, 'T24': 642.0}),

    ("Motor Bravo — Desvio de Parâmetros e Termodinâmica Irregular",
     {'T30': 1604.0, 'T50': 1418.0, 'P30': 540.0, 'phi': 532.0,
      'Nf': 2379.0, 'Nc': 9025.0, 'Ps30': 45.9, 'NRf': 2379.0,
      'NRc': 8117.0, 'BPR': 8.47, 'htBleed': 403.0,
      'W31': 40.2, 'W32': 24.2, 'T24': 643.0}),
]

for nome, leituras in casos:
    resultado = diagnosticar_motor(leituras)
    print(f"\n  {nome}")
    print(f"    T30={leituras.get('T30')}°R | T50={leituras.get('T50')}°R | P30={leituras.get('P30')} psia")
    print(f"    {resultado['icone']} {resultado['status']}  (Probabilidade de risco: {resultado['risco']})")

print("\n" + "=" * 60)

# ════════════════════════════════════════════════════════════
# 8. RESUMO EXECUTIVO (PRONTO PARA O LINKEDIN)
# ════════════════════════════════════════════════════════════
print(
    f"""
Resumo Executivo
----------------
Motores monitorados : {train_df['unit'].nunique()}
Ciclos analisados   : {len(train_df):,}
ROC-AUC             : {auc:.4f}
Acurácia            : {acc:.2%}

Objetivo:
Detectar degradação de motores turbofan utilizando dados históricos de sensores, 
simulando sistemas modernos de manutenção preditiva da indústria aeronáutica.
"""
)
print("=" * 60)
print("  Projeto concluído com sucesso. Bons voos ✈")
print("=" * 60)