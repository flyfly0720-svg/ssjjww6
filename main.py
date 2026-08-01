
# -*- coding: utf-8 -*-
"""
비만·신체활동과 암 발병의 관계: Cox 회귀분석 + YAP/TAZ 미분방정식 모델
================================================================

[앱 구성]
탭 1. Cox 비례위험회귀분석 (실제 국민건강영양조사 HN24 자료)
탭 2. YAP/TAZ 세포 신호 미분방정식 모델 (파라미터 조절 가능)
탭 3. 역방향 디버깅 — 통계 결과로부터 미분방정식의 문제를 역으로 추적

[배포 안정성 설계]
Cox 회귀와 미분방정식 계산 모두 numpy만으로 직접 구현되어 있어
(cox_model.py, yap_taz_model.py), lifelines나 scipy 설치가 실패해도
이 앱은 영향을 받지 않는다. 이 파일이 요구하는 외부 패키지는
streamlit, pandas, numpy, plotly 4개뿐이다.

데이터 출처: 질병관리청 국민건강영양조사 제9기 2차년도(2024, HN24) 원시자료
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from cox_model import fit_cox, summary_table
from yap_taz_model import simulate, steady_state_Y

st.set_page_config(
    page_title="비만·운동과 암: Cox 회귀 + YAP/TAZ 모델",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv("hn24_cox_data.csv")
    return df


df = load_data()

st.title("비만·신체활동이 암 발병 위험에 미치는 영향")
st.caption("국민건강영양조사(HN24, 2024) 원시자료 기반 Cox 회귀분석 + YAP/TAZ 세포 신호 미분방정식 모델링")

tab1, tab2, tab3 = st.tabs([
    "1. Cox 비례위험회귀분석",
    "2. YAP/TAZ 미분방정식 모델",
    "3. 역방향 디버깅 (통계 → 메커니즘)",
])

# ==================================================================
# 탭 1: Cox 비례위험회귀분석
# ==================================================================
with tab1:
    st.header("Cox 비례위험회귀분석 (Cox Proportional Hazards Regression)")

    st.markdown(r"""
**왜 로지스틱 회귀 대신 Cox 회귀인가?**

로지스틱 회귀는 "암 진단을 받았는가(0 또는 1)"만 사용하고,
**몇 살에 진단받았는지**, 그리고 **비진단자가 몇 살까지 관찰되었는지**라는
시간 정보를 버립니다. 예를 들어 현재 79세인데 암이 없는 사람과 20세인데
암이 없는 사람은 로지스틱 회귀에서 똑같이 "0"으로 취급되지만, 두 사람이
가진 위험 정보의 양은 전혀 다릅니다.

Cox 회귀는 각 개인의 **관찰기간(time)**과 **사건 발생 여부(event)**를 함께
사용해서 이 문제를 해결합니다.

- 암 진단자: `event=1`, `time=진단 시 나이`
- 암 비진단자: `event=0`, `time=현재 나이` → **중도절단(censoring)**:
  "이 나이까지는 최소한 암이 없었다"는 정보만 확실하고, 그 이후는 알 수 없다는 뜻
""")

    with st.expander("Cox 모형의 수학적 구조 보기"):
        st.markdown(r"""
Cox 모형은 시간 $t$에서의 위험함수(hazard function) $h(t)$를 다음과 같이 모델링합니다.

$$
h(t \mid X) = h_0(t) \cdot \exp(\beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p)
$$

- $h_0(t)$: 기저위험함수(baseline hazard) — 모든 독립변수가 0일 때의 위험도. 시간에 따라 변할 수 있음.
- $\exp(\beta_i)$: **위험비(Hazard Ratio, HR)** — 변수 $X_i$가 1단위 증가할 때 위험도가 몇 배가 되는지.
- 로지스틱의 오즈비(OR)와 비슷하지만, OR은 "특정 시점의 정적인 승산"이고
  HR은 "매 순간의 위험 속도(위험률)"라는 점에서 다릅니다.

Cox 모형이 "비례위험(proportional hazards)"이라 불리는 이유는, $h_0(t)$가 시간에 따라
어떻게 변하든 상관없이 두 사람의 위험비 $h(t|X_1)/h(t|X_2)$는 시간에 걸쳐 **항상 일정하다**고
가정하기 때문입니다. 이 가정 덕분에 $h_0(t)$의 구체적 형태를 몰라도 $\beta$를
추정할 수 있습니다 (부분우도, partial likelihood — 이 앱에서는 numpy로 직접 구현했습니다).
""")

    st.subheader("분석 대상 데이터")
    col1, col2, col3 = st.columns(3)
    col1.metric("전체 표본 수", f"{len(df):,}명")
    col2.metric("암 진단(사건 발생)", f"{int(df['event'].sum()):,}명")
    col3.metric("중도절단(비진단)", f"{int((df['event']==0).sum()):,}명")

    st.markdown("**분석에 포함할 변수를 선택하세요.**")
    covariate_options = {
        "BMI": "체질량지수 (BMI, kg/m²)",
        "aerobic_pa": "유산소 신체활동 실천율 (0=미실천, 1=실천)",
        "fasting_glucose": "공복혈당 (mg/dL)",
        "age": "나이",
        "female": "성별 (0=남성, 1=여성)",
    }
    available_covs = [c for c in covariate_options if c in df.columns]
    default_covs = [c for c in ["BMI", "aerobic_pa", "fasting_glucose", "female"] if c in available_covs]

    selected = st.multiselect(
        "독립변수(공변량) 선택",
        options=available_covs,
        default=default_covs,
        format_func=lambda x: covariate_options[x],
    )

    if len(selected) == 0:
        st.warning("최소 1개 이상의 변수를 선택하세요.")
    else:
        cox_df = df[["time", "event"] + selected].dropna()
        X = cox_df[selected].values.astype(float)
        # 수치안정성을 위해 표준화(평균 0, 표준편차 1)한 뒤 적합하고,
        # 계수는 다시 원래 스케일로 환산해서 보여준다.
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_scaled = (X - X_mean) / X_std

        time_arr = cox_df["time"].values.astype(float)
        event_arr = cox_df["event"].values.astype(float)

        beta_scaled, se_scaled, loglik = fit_cox(X_scaled, time_arr, event_arr)
        # 원래 스케일로 환산: beta_original = beta_scaled / std
        beta = beta_scaled / X_std
        se = se_scaled / X_std

        rows = summary_table(beta, se, [covariate_options[c] for c in selected])
        summary_df = pd.DataFrame(rows)
        summary_display = summary_df[["변수", "beta", "HR", "CI_low", "CI_high", "p"]].copy()
        summary_display.columns = ["변수", "회귀계수(β)", "위험비(HR)", "HR 95% 하한", "HR 95% 상한", "p-value"]
        summary_display = summary_display.round(4)

        st.subheader("위험비(Hazard Ratio) 결과")
        st.dataframe(summary_display, width='stretch')

        # ---- Plotly: 위험비 forest plot ----
        hr_vals = summary_df["HR"].values
        lower = summary_df["CI_low"].values
        upper = summary_df["CI_high"].values
        labels = summary_df["변수"].tolist()

        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(
            x=hr_vals, y=labels,
            mode="markers",
            marker=dict(size=12, color="crimson"),
            error_x=dict(
                type="data", symmetric=False,
                array=upper - hr_vals, arrayminus=hr_vals - lower,
                thickness=2, width=6,
            ),
            name="위험비(HR)",
        ))
        fig_hr.add_vline(x=1, line_dash="dash", line_color="gray",
                          annotation_text="HR=1 (영향 없음 기준선)")
        fig_hr.update_layout(
            title="변수별 위험비(HR)와 95% 신뢰구간",
            xaxis_title="위험비 (Hazard Ratio)",
            yaxis_title="",
            height=120 + 60 * len(labels),
            template="plotly_white",
        )
        st.plotly_chart(fig_hr, width='stretch')

        st.markdown("""
**해석 방법**: 점이 기준선(HR=1)보다 오른쪽에 있고 오차막대가 1을 넘지 않으면
"위험을 유의하게 높이는 변수", 왼쪽에 있으면 "위험을 유의하게 낮추는 변수"입니다.
오차막대가 1을 가로지르면 통계적으로 유의하지 않다는 뜻입니다.
""")

        # ---- 카플란-마이어 생존곡선 (신체활동 유무별, numpy로 직접 계산) ----
        if "aerobic_pa" in df.columns:
            st.subheader("암 무진단 생존확률 (신체활동 실천 여부별)")
            st.markdown("""
Cox 모형이 '평균적인 위험 배수'를 알려준다면, 아래 곡선(카플란-마이어 추정법)은
나이가 들수록 '암이 없을 확률(무진단 생존 확률)'이 실제로 어떻게 떨어지는지를
그대로 보여줍니다.
""")

            def kaplan_meier(time_g, event_g):
                """
                카플란-마이어 생존함수 추정치를 numpy로 직접 계산한다.

                원리: 각 사건 발생 시점 t_k에서, 그 시점 직전까지 생존해있던
                사람 수(n_k)와 그 시점에 사건이 발생한 사람 수(d_k)를 이용해
                S(t_k) = S(t_{k-1}) * (1 - d_k/n_k) 로 생존확률을 갱신한다.
                즉 "각 시점을 무사히 넘길 조건부 확률"을 계속 곱해나가는 방식이다.
                """
                order = np.argsort(time_g)
                t_sorted = time_g[order]
                e_sorted = event_g[order]
                unique_times = np.unique(t_sorted[e_sorted == 1])
                surv = 1.0
                surv_curve = [(0, 1.0)]
                n_total = len(t_sorted)
                for tk in unique_times:
                    n_at_risk = np.sum(t_sorted >= tk)
                    d_events = np.sum((t_sorted == tk) & (e_sorted == 1))
                    surv *= (1 - d_events / n_at_risk)
                    surv_curve.append((tk, surv))
                return surv_curve

            fig_km = go.Figure()
            for val, name, color in [(0, "신체활동 미실천", "orangered"), (1, "신체활동 실천", "royalblue")]:
                mask = df["aerobic_pa"] == val
                curve = kaplan_meier(df.loc[mask, "time"].values.astype(float),
                                      df.loc[mask, "event"].values.astype(float))
                xs = [c[0] for c in curve]
                ys = [c[1] for c in curve]
                fig_km.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", name=name,
                    line=dict(color=color, width=2, shape="hv"),
                ))
            fig_km.update_layout(
                title="나이에 따른 암 무진단 생존확률",
                xaxis_title="나이",
                yaxis_title="암 무진단 생존확률",
                template="plotly_white",
                height=450,
            )
            st.plotly_chart(fig_km, width='stretch')

        st.session_state["cox_summary_df"] = summary_df
        st.session_state["cox_covariates"] = selected

# ==================================================================
# 탭 2: YAP/TAZ 미분방정식 모델
# ==================================================================
with tab2:
    st.header("YAP/TAZ 세포 신호 미분방정식 모델")

    st.markdown(r"""
Cox 회귀가 "인구 집단 수준에서 비만·운동이 암 진단 위험과 통계적으로
얼마나 연관되는가"를 보여준다면, 이 미분방정식은 "세포 내부에서
비만·운동 신호가 실제로 어떤 분자 경로를 통해 암세포 증식에 영향을
주는가"라는 **메커니즘**을 시뮬레이션합니다.

**핵심 배경 지식 (YAP/TAZ)**

YAP(Yes-associated protein)/TAZ는 Hippo 신호경로의 핵심 전사공동활성인자입니다.
인산화되어 세포질에 갇히면 비활성 상태이고, 탈인산화되어 **핵으로 이동**하면
TEAD 전사인자와 결합해 세포증식·항세포사멸 유전자의 발현을 촉진하는
종양유발(oncogenic) 신호로 작동합니다.

- **Akt (PI3K-Akt 경로)**: 비만·고혈당으로 인한 인슐린/IGF-1 신호가 과활성화되면
  Akt가 활성화되고, 이는 YAP/TAZ의 핵 내 이동과 안정화를 촉진합니다
  (Hippo 경로를 억제하는 방향).
- **AMPK**: 운동으로 인한 세포 에너지 소모(ATP 감소)를 감지하는 센서로,
  활성화되면 LATS1/2 키나아제를 거쳐 YAP를 직접 인산화(Ser94 부위)하여
  세포질에 가두고 분해를 촉진합니다 (Hippo 경로를 활성화하는 방향, YAP 억제).
""")

    st.subheader("모델 방정식")
    st.latex(r"\frac{dY}{dt} = \alpha \cdot A - \beta \cdot M \cdot Y - \gamma \cdot Y")
    st.latex(r"\frac{dC}{dt} = r \cdot C\left(1 - \frac{C}{K}\right) + \delta \cdot Y \cdot C")

    st.markdown("""
- $Y(t)$: 시간 $t$에서 핵 내 활성형 YAP/TAZ 농도
- $C(t)$: 시간 $t$에서 변이 암세포 수(상대적 종양 크기)
""")

    with st.expander("각 항이 왜 이렇게 생겼는지 — 단계별 설명"):
        st.markdown(r"""
**① $dY/dt$ 식 — YAP/TAZ 농도가 시간에 따라 어떻게 변하는가**

| 항 | 부호 | 의미 |
|---|---|---|
| $\alpha \cdot A$ | + | Akt 신호($A$)에 비례해 YAP/TAZ가 새로 생성·안정화되는 항. $A$가 클수록(비만·고혈당이 심할수록) YAP 농도가 늘어나는 속도가 커집니다. |
| $-\beta \cdot M \cdot Y$ | − | AMPK 신호($M$)가 **현재 존재하는 YAP 농도($Y$)에 비례해서** YAP를 인산화·분해시키는 항입니다. $M$과 $Y$의 곱으로 쓴 이유는, 화학반응에서 두 물질(효소인 AMPK와 기질인 YAP)이 만나야 반응이 일어난다는 질량작용 법칙(mass-action law)을 반영하기 때문입니다. 운동을 아무리 많이 해도($M$이 커도) YAP가 애초에 적으면($Y\approx0$) 분해될 게 없으므로 이 항도 작아집니다. |
| $-\gamma \cdot Y$ | − | 신호와 무관하게 일어나는 YAP의 자연분해(기저 turnover)입니다. 모든 단백질은 시간이 지나면 자연히 분해되므로, 농도에 비례해 사라지는 1차 반응(first-order decay)으로 표현합니다. |

**② $dC/dt$ 식 — 암세포 수가 시간에 따라 어떻게 변하는가**

| 항 | 의미 |
|---|---|
| $r \cdot C\left(1-\dfrac{C}{K}\right)$ | 로지스틱 증식항. 암세포는 초기($C \ll K$)에는 거의 지수적으로($r\cdot C$) 증식하지만, 혈액공급·공간 등 자원의 한계 $K$(수용용량, carrying capacity)에 가까워질수록 $(1-C/K)$ 항이 0에 가까워지면서 증식이 자연히 둔화됩니다. 개체군생태학에서 쓰는 표준 로지스틱 성장모형과 수학적으로 동일한 구조입니다. |
| $\delta \cdot Y \cdot C$ | YAP/TAZ 신호가 암세포 증식을 **추가로 가속**시키는 항. YAP는 세포증식·항세포사멸 유전자를 전사활성화하므로, YAP 농도($Y$)가 높을수록 이미 있는 암세포 수($C$)에 비례해서 증식 속도가 더 빨라집니다. 여기서도 $Y$와 $C$의 곱으로 쓴 이유는 ①과 같은 질량작용 법칙입니다 — YAP 신호가 암세포에 "작용"하려면 둘 다 존재해야 하기 때문입니다. |

**③ 정상상태(steady-state) 해석**

$dY/dt=0$으로 두고 풀면:
""")
        st.latex(r"Y^{*} = \frac{\alpha A}{\beta M + \gamma}")
        st.markdown(r"""
- $M=0$(운동을 전혀 하지 않을 때): $Y^{*} = \dfrac{\alpha A}{\gamma}$로 **최댓값**을 가짐 → 암세포 가속 증식 항($\delta Y^* C$)도 최대가 됨
- $M$이 커질수록(운동을 많이 할수록): 분모($\beta M + \gamma$)가 커져 $Y^{*}$는 **단조감소** → 암세포 가속 증식 효과가 줄어듦

즉 이 모형이 "운동이 암 위험을 낮출 수 있다"고 예측하는 수학적 근거는,
AMPK($M$)가 YAP의 정상상태 농도를 분모에서 직접 낮추기 때문입니다.
""")

    st.subheader("파라미터 조절")
    st.markdown("아래 슬라이더로 각 파라미터를 바꿔가며 $Y(t)$, $C(t)$의 변화를 관찰해보세요.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**세포 신호 관련 파라미터**")
        alpha = st.slider(
            "α (알파) — Akt 신호의 YAP 생성 효율",
            min_value=0.0, max_value=3.0, value=1.0, step=0.05,
            help="Akt 활성 신호(A) 한 단위가 YAP/TAZ 생성 속도로 얼마나 잘 "
                 "전환되는지를 나타내는 비례상수입니다. 값이 클수록 같은 비만/고혈당 "
                 "수준(A)에서도 YAP가 더 빨리 축적됩니다. 생물학적으로는 PI3K-Akt "
                 "경로의 신호전달 효율(수용체 민감도, 하위 인산화효소 활성 등)에 해당합니다."
        )
        beta = st.slider(
            "β (베타) — AMPK의 YAP 분해 효율",
            min_value=0.0, max_value=3.0, value=0.5, step=0.05,
            help="AMPK 활성 신호(M)가 YAP를 인산화·분해시키는 반응의 효율(반응속도상수)입니다. "
                 "값이 클수록 같은 운동량(M)으로도 YAP를 더 효과적으로 억제합니다. "
                 "생물학적으로는 AMPK가 LATS1/2를 거쳐 YAP Ser94를 인산화하는 "
                 "키나아제 반응의 촉매 효율에 해당합니다."
        )
        gamma = st.slider(
            "γ (감마) — YAP의 자연분해율",
            min_value=0.01, max_value=2.0, value=0.2, step=0.01,
            help="Akt·AMPK 신호와 무관하게 일어나는 YAP 단백질의 기저 분해율입니다. "
                 "값이 클수록 YAP의 세포 내 반감기가 짧아져, 다른 신호가 없어도 "
                 "농도가 빨리 낮은 수준으로 돌아갑니다."
        )
        delta = st.slider(
            "δ (델타) — YAP의 암세포 증식 촉진 효율",
            min_value=0.0, max_value=0.1, value=0.02, step=0.002,
            help="YAP 농도 한 단위가 암세포 증식 속도를 얼마나 가속시키는지 나타내는 "
                 "비례상수입니다. 값이 클수록 같은 YAP 농도에서도 종양이 더 빠르게 "
                 "자랍니다. 생물학적으로는 YAP-TEAD 복합체가 세포증식 유전자(CTGF, CYR61 등)를 "
                 "얼마나 강하게 전사활성화하는지에 해당합니다."
        )

    with col_b:
        st.markdown("**개인 특성 및 종양 성장 관련 파라미터**")
        A = st.slider(
            "A — 비만/고혈당 유래 Akt 활성 강도",
            min_value=0.0, max_value=3.0, value=1.0, step=0.05,
            help="개인의 비만도·공복혈당 수준을 반영하는 입력값입니다. BMI가 높거나 "
                 "공복혈당이 높을수록(인슐린 저항성이 클수록) 이 값이 커진다고 볼 수 "
                 "있습니다. Cox 회귀분석 탭에서 얻은 BMI, fasting_glucose의 위험비가 "
                 "클수록 이 파라미터를 더 크게 설정하는 것이 합리적입니다."
        )
        M = st.slider(
            "M — 운동 유래 AMPK 활성 강도",
            min_value=0.0, max_value=3.0, value=0.5, step=0.05,
            help="개인의 신체활동(운동) 수준을 반영하는 입력값입니다. 유산소 운동을 "
                 "많이 할수록 골격근에서 AMPK가 더 강하게, 더 오래 활성화되므로 "
                 "이 값이 커진다고 볼 수 있습니다. Cox 회귀분석 탭의 aerobic_pa 변수와 "
                 "개념적으로 대응됩니다."
        )
        r = st.slider(
            "r — 암세포 기저 증식률",
            min_value=0.0, max_value=1.0, value=0.3, step=0.01,
            help="YAP/TAZ 신호와 무관하게 암세포가 갖는 고유한 증식 속도입니다. "
                 "암종(cancer type)마다 다른 세포주기 길이, 돌연변이 종류에 따라 "
                 "달라질 수 있는 값입니다."
        )
        K = st.slider(
            "K — 종양의 수용용량(carrying capacity)",
            min_value=10, max_value=500, value=100, step=10,
            help="해당 조직·장기에서 혈액공급, 물리적 공간 등의 제약으로 암세포가 "
                 "도달할 수 있는 최대 크기입니다. 로지스틱 성장모형의 표준 개념으로, "
                 "값이 클수록 암세포가 더 오래, 더 크게 자랄 수 있는 여지가 생깁니다."
        )
        Y0 = st.slider("초기 YAP 농도 Y(0)", 0.0, 2.0, 0.1, 0.05)
        C0 = st.slider("초기 암세포 수 C(0)", 0.1, 10.0, 1.0, 0.1)
        t_max = st.slider("시뮬레이션 총 시간", 10, 200, 50, 10)

    params = dict(alpha=alpha, beta=beta, gamma=gamma, r=r, K=K, delta=delta, A=A, M=M)
    t, Y, C = simulate(params, t_max=t_max, Y0=Y0, C0=C0)
    Y_star = steady_state_Y(alpha, A, beta, M, gamma)

    col1, col2 = st.columns(2)
    col1.metric("YAP 정상상태 농도 Y*", f"{Y_star:.3f}")
    col2.metric(f"t={t_max}에서 암세포 수 C(t)", f"{C[-1]:.2f}")

    fig_ode = go.Figure()
    fig_ode.add_trace(go.Scatter(x=t, y=Y, mode="lines", name="Y(t) — 활성형 YAP/TAZ 농도",
                                  line=dict(color="darkorange", width=2), yaxis="y1"))
    fig_ode.add_trace(go.Scatter(x=t, y=C, mode="lines", name="C(t) — 암세포 수",
                                  line=dict(color="firebrick", width=2), yaxis="y2"))
    fig_ode.update_layout(
        title="시간에 따른 YAP/TAZ 농도와 암세포 수의 변화",
        xaxis_title="시간 t",
        yaxis=dict(title=dict(text="YAP/TAZ 농도 Y(t)", font=dict(color="darkorange")),
                   tickfont=dict(color="darkorange")),
        yaxis2=dict(title=dict(text="암세포 수 C(t)", font=dict(color="firebrick")),
                    tickfont=dict(color="firebrick"), overlaying="y", side="right"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )
    st.plotly_chart(fig_ode, width='stretch')

    st.subheader("운동 강도(M)에 따른 YAP 정상상태 농도 변화")
    M_range = np.linspace(0, 3, 100)
    Y_star_range = steady_state_Y(alpha, A, beta, M_range, gamma)
    fig_mstar = go.Figure()
    fig_mstar.add_trace(go.Scatter(x=M_range, y=Y_star_range, mode="lines",
                                    line=dict(color="teal", width=3)))
    fig_mstar.add_vline(x=M, line_dash="dash", line_color="gray",
                         annotation_text=f"현재 설정 M={M}")
    fig_mstar.update_layout(
        title="Y* = αA / (βM + γ) — 운동 강도가 커질수록 YAP 정상상태가 감소",
        xaxis_title="운동 강도 M (AMPK 활성)",
        yaxis_title="YAP 정상상태 농도 Y*",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_mstar, width='stretch')

    st.session_state["ode_params"] = params

# ==================================================================
# 탭 3: 역방향 디버깅 — 통계 결과로부터 미분방정식 파라미터 역추정
# ==================================================================
with tab3:
    st.header("역방향 디버깅: Cox 회귀 결과로 미분방정식의 문제를 역으로 추적하기")

    st.markdown("""
지금까지 탭 1(통계)과 탭 2(메커니즘 모델)를 따로 살펴봤습니다. 이 탭에서는
**두 결과가 서로 모순될 때, 그 모순을 실마리 삼아 어느 쪽(또는 어떤 가정)이
문제인지를 거꾸로 추적**하는 과정을 보여줍니다. 이게 "디버깅"이라 부르는 이유는,
소프트웨어에서 예상과 다른 출력이 나왔을 때 원인을 거슬러 올라가 찾는 과정과
논리 구조가 완전히 같기 때문입니다.
""")

    if "cox_summary_df" not in st.session_state:
        st.warning("먼저 '1. Cox 비례위험회귀분석' 탭에서 분석을 실행해 주세요.")
    else:
        summary_df = st.session_state["cox_summary_df"]

        st.subheader("1단계 — 실제 데이터의 Cox 회귀 결과 확인")
        st.markdown("""
미분방정식 모델이 예측하는 방향은 다음과 같습니다.

> "운동 강도 M이 커지면 YAP 정상상태 Y*가 감소하고, 그 결과
> 암세포 증식 속도(δY*C)도 감소한다. 즉 **운동은 암 위험을 낮춰야 한다**
> (위험비 HR < 1)."

그렇다면 실제 Cox 회귀 결과에서 신체활동 실천율의 위험비는
이 예측과 일치할까요?
""")

        pa_row = summary_df[summary_df["변수"].str.contains("신체활동", na=False)]
        if len(pa_row) > 0:
            hr_pa = pa_row["HR"].values[0]
            p_pa = pa_row["p"].values[0]
            lower_pa = pa_row["CI_low"].values[0]
            upper_pa = pa_row["CI_high"].values[0]

            st.metric("신체활동 실천의 위험비(HR)", f"{hr_pa:.3f}",
                      delta=f"p={p_pa:.4f}", delta_color="off")

            if hr_pa > 1 and p_pa < 0.05:
                st.error(f"""
**모순 발견**: HR = {hr_pa:.3f} (95% CI {lower_pa:.3f}–{upper_pa:.3f}), p = {p_pa:.4f}로
통계적으로 유의합니다. 그런데 HR이 1보다 **크다**는 것은 "운동을 실천하는
사람일수록 암 진단 위험이 더 높다"는 뜻으로, 미분방정식 모델이 예측한
방향(운동 → 암 위험 감소)과 **정반대**입니다.
""")
            elif hr_pa < 1 and p_pa < 0.05:
                st.success(f"""
HR = {hr_pa:.3f} (95% CI {lower_pa:.3f}–{upper_pa:.3f}), p = {p_pa:.4f}로
통계적으로 유의하며, 미분방정식 모델의 예측(운동 → 암 위험 감소, HR<1)과
방향이 **일치**합니다.
""")
            else:
                st.info(f"""
HR = {hr_pa:.3f} (95% CI {lower_pa:.3f}–{upper_pa:.3f}), p = {p_pa:.4f}로
통계적으로 유의하지 않습니다. 표본 내에서 방향성 자체를 확정하기 어려운
상태이며, 이 경우도 아래 디버깅 논리를 그대로 적용해 원인을 따져볼 수 있습니다.
""")
        else:
            st.info("탭 1에서 '유산소 신체활동 실천율' 변수를 선택하면 이 비교를 볼 수 있습니다.")

        st.subheader("2단계 — 모순의 원인을 거슬러 추적하기 (디버깅 트리)")
        st.markdown("""
소프트웨어 디버깅에서 "출력이 예상과 다르면 입력, 로직, 가정을 하나씩 되짚어본다"는
원칙을 그대로 적용해 보겠습니다. 가능한 원인을 **모델이 맞고 데이터 해석이 틀렸을
경우**와 **데이터가 맞고 모델의 가정이 틀렸을 경우**로 나눠 점검합니다.
""")

        with st.expander("원인 후보 ① — 역인과관계(reverse causation): 왜 이게 생기는가"):
            st.markdown("""
**왜 이 문제가 생기는가**: 이 자료는 **단면조사(cross-sectional)**입니다. 즉
"신체활동 수준"과 "암 진단 여부"를 **같은 시점(2024년)**에 한 번만 측정합니다.
그런데 실제로 시간 순서는 다음과 같을 수 있습니다.

1. 어떤 사람이 암 진단을 받음
2. 진단 이후 건강관리를 위해 걷기·산책 등 신체활동을 새로 시작하거나 늘림
3. 2024년 설문 시점에는 "신체활동 실천=1"로 기록됨

이러면 원인(암)과 결과(운동)의 시간 순서가 뒤바뀐 채로 통계에 들어가서,
실제로는 "암이 운동을 유발"했는데 모형은 이를 "운동이 암을 유발"한 것처럼
계산하게 됩니다. Cox 모형에 진단 나이(time)를 넣었더라도, **독립변수인
운동 수준 자체는 진단 시점이 아니라 2024년 현재 시점의 값**이라는 근본적
한계가 남아있기 때문에 이 문제가 완전히 해결되지 않습니다.

**왜 미분방정식은 이걸 못 잡아내는가**: 미분방정식 모델은 M(운동)이
**독립적인 입력(외생변수)**이라고 가정합니다. 즉 M이 커져서 Y가
줄어드는 인과 방향만 모델링했지, C(암세포 수)가 거꾸로 M에 영향을
주는 피드백 경로(C → M)는 애초에 방정식에 없습니다. 모델 자체가
"운동은 원인, 암은 결과"라는 가정을 전제로 만들어졌기 때문에, 데이터에서
반대 방향의 인과가 섞여 있어도 모형 구조상 감지할 수 없습니다.
""")

        with st.expander("원인 후보 ② — 건강검진 접근성에 의한 교란(confounding)"):
            st.markdown("""
**왜 이 문제가 생기는가**: 신체활동을 꾸준히 하는 사람은 대체로 건강에 관심이
많아 건강검진도 더 자주 받는 경향이 있습니다. 검진을 자주 받을수록 무증상
초기 암도 더 잘 **발견**됩니다. 즉 "운동을 하는 사람"과 "암이 새로 생기는
사람"이 아니라, "운동을 하는 사람"과 "암을 (더 일찍) **발견하는** 사람"이
연결되어 있을 수 있습니다. 이 경우 실제 암 발생률 차이가 없어도 관측된
진단율은 차이가 나게 됩니다.

**미분방정식에서는 어디에 해당하는가**: 이 모델의 C(t)는 "실제 존재하는
암세포 수"를 나타내지 "진단된 암세포 수"를 나타내지 않습니다. 즉 모델의
C와 Cox 회귀의 종속변수(암 **진단** 여부)가 개념적으로 다른 대상을
가리키고 있을 수 있다는 뜻입니다. 이 괴리가 크면, 모델이 아무리 정교해도
관측 데이터와 맞지 않는 게 당연합니다.
""")

        with st.expander("원인 후보 ③ — 표본 내 사건 수 부족(power 문제)"):
            n_event = int(df["event"].sum())
            st.markdown(f"""
**왜 이 문제가 생기는가**: 이번 분석에서 암 진단(사건) 수는 **{n_event}건**으로,
전체 표본({len(df):,}명) 대비 비율이 크지 않습니다. Cox 회귀 같은 사건기반 모형은
사건 수가 적을수록 추정치가 불안정해지고 우연에 의한 변동(표본오차)의 영향을
크게 받습니다. 즉 지금 관측된 HR>1이 실제 위험 증가를 반영하는 게 아니라,
사건 수가 적어서 생긴 통계적 잡음(noise)일 가능성도 배제할 수 없습니다.

**확인 방법**: 신뢰구간의 폭을 보면 됩니다. 위 1단계에서 확인한 95% 신뢰구간이
넓을수록(하한과 상한의 차이가 클수록) 추정이 불안정하다는 뜻이고, 이는 표본
크기·사건 수 부족을 시사하는 신호입니다.
""")

        st.subheader("3단계 — 미분방정식 파라미터를 역으로 추정해보기")
        st.markdown(r"""
지금까지 정성적으로 원인을 짚어봤다면, 이번에는 정량적으로 접근해 봅니다.
"만약 관측된 Cox 회귀의 HR 방향이 실제 생물학적 현상을 그대로 반영한다고
**가정**하면, 미분방정식의 파라미터는 어떤 조건을 만족해야 하는가"를 역으로
풀어보는 것입니다. 이렇게 하면 "모델의 어느 파라미터 가정이 비현실적이었는지"를
구체적으로 짚어낼 수 있습니다.

모델이 원래 가정한 Y* = αA / (βM + γ)는 M에 대해
**항상 감소**하는 함수입니다(β, γ > 0일 때 수학적으로 증가할 수 없음).
따라서 이 함수 형태를 유지하는 한, 어떤 α, β, γ 조합을 넣어도
"운동이 암 위험을 높인다"는 관측 결과를 재현할 수 없습니다.

이는 파라미터 값이 잘못된 게 아니라, **모델의 함수 형태(구조) 자체가 관측된
현상을 표현할 수 없다**는 뜻입니다. 아래에서 실제로 파라미터를 바꿔가며
이를 확인해볼 수 있습니다.
""")

        if "ode_params" in st.session_state:
            p = st.session_state["ode_params"]
            M_test = np.linspace(0.01, 3, 200)
            Y_test = steady_state_Y(p["alpha"], p["A"], p["beta"], M_test, p["gamma"])
            slope_sign = np.sign(np.diff(Y_test))
            always_decreasing = np.all(slope_sign <= 0)

            fig_debug = go.Figure()
            fig_debug.add_trace(go.Scatter(x=M_test, y=Y_test, mode="lines",
                                            line=dict(color="purple", width=3),
                                            name="모델이 예측하는 Y*(M)"))
            fig_debug.update_layout(
                title="현재 파라미터에서 Y*(M) 함수의 모양 — 항상 감소하는가?",
                xaxis_title="운동 강도 M",
                yaxis_title="YAP 정상상태 Y*",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig_debug, width='stretch')

            if always_decreasing:
                st.warning("""
**진단 결과**: 탭 2에서 설정한 현재 파라미터로는 Y*(M)가 어떤 값을
넣어도 항상 감소합니다. 즉 이 함수 형태로는 "운동↑ → 암 위험↑"이라는
관측된 패턴을 절대 만들어낼 수 없습니다. → **모델 구조를 바꿔야 한다는
신호**입니다. (예: M이 매개하는 다른 경로, 역인과 항, 또는 검진율 항을
추가하는 방향의 모델 확장이 필요합니다.)
""")
        else:
            st.info("탭 2에서 파라미터를 한 번 조절하면 이 그래프가 나타납니다.")

        st.markdown("""
**요약**: 미분방정식 모델(탭 2)은 "운동 → AMPK↑ → YAP↓ → 암 위험↓"이라는
**하나의 가능한 메커니즘**을 수학적으로 정교하게 표현한 것이지, 실제 관측
데이터를 있는 그대로 재현하도록 맞춰진 모델이 아닙니다. Cox 회귀(탭 1)와
비교했을 때 방향이 어긋난다면, 그 자체가 "이 데이터는 단면조사이므로
역인과·교란 가능성을 배제할 수 없다"는 탐구의 핵심 논점이 됩니다. 즉
모형과 통계가 불일치하는 지점을 찾아내는 것 자체가 이 탐구의 중요한
결론 중 하나로 보고서에 쓸 수 있습니다.
""")
