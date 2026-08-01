
# -*- coding: utf-8 -*-
"""
YAP/TAZ - 암세포 동역학 미분방정식 모델 (numpy 전용, 외부 수치적분 라이브러리 불필요)

[생물학적 배경]
YAP(Yes-associated protein)/TAZ는 Hippo 신호경로의 핵심 전사공동활성인자다.
인산화되어 세포질에 갇히면 비활성 상태이고, 탈인산화되어 핵으로 이동하면
TEAD 전사인자와 결합해 세포증식·항세포사멸 유전자의 발현을 촉진하는
종양유발(oncogenic) 신호로 작동한다.

- Akt (PI3K-Akt 경로): 비만·고혈당으로 인한 인슐린/IGF-1 신호가 과활성화되면
  Akt가 활성화되고, 이는 YAP/TAZ의 핵 내 이동과 안정화를 촉진한다
  (Hippo 경로를 억제하는 방향).
- AMPK: 운동으로 인한 세포 에너지 소모(ATP 감소)를 감지하는 센서로,
  활성화되면 LATS1/2 키나아제를 거쳐 YAP를 직접 인산화(Ser94 부위)하여
  세포질에 가두고 분해를 촉진한다 (Hippo 경로를 활성화하는 방향, YAP 억제).

[모델 방정식]
dY/dt = alpha*A - beta*M*Y - gamma*Y
dC/dt = r*C*(1 - C/K) + delta*Y*C

Y(t): 시간 t에서 핵 내 활성형 YAP/TAZ 농도
C(t): 시간 t에서 변이 암세포 수(상대적 종양 크기)
A   : 비만/고혈당 유래 Akt 활성 상수 (YAP 생성·안정화 촉진)
M   : 운동 유래 AMPK 활성 상수 (YAP 인산화·분해 촉진)
"""

import numpy as np


def yap_taz_odes(state, alpha, beta, gamma, r, K, delta, A, M):
    """
    YAP/TAZ - 암세포 동역학 연립미분방정식의 우변(도함수)을 계산한다.

    각 항의 의미
    ---------------------------------------------------------------
    dY/dt = alpha*A - beta*M*Y - gamma*Y
      + alpha*A   : Akt 신호(A)에 비례해 YAP/TAZ가 새로 생성·안정화되는 항.
                    A가 클수록(비만·고혈당이 심할수록) YAP 농도가 늘어나는
                    속도가 커진다.
      - beta*M*Y  : AMPK 신호(M)가 현재 존재하는 YAP 농도(Y)에 비례해서
                    YAP를 인산화·분해시키는 항. M과 Y의 곱으로 쓰는 이유는,
                    화학반응에서 두 물질(효소 AMPK와 기질 YAP)이 만나야
                    반응이 일어난다는 질량작용 법칙(mass-action law) 때문이다.
                    운동을 아무리 많이 해도(M이 커도) YAP가 애초에 적으면
                    (Y≈0) 분해될 게 없으므로 이 항도 작아진다.
      - gamma*Y   : 신호와 무관하게 일어나는 YAP의 자연분해(기저 turnover).
                    모든 단백질은 시간이 지나면 자연히 분해되므로, 농도에
                    비례해 사라지는 1차 반응(first-order decay)으로 표현한다.

    dC/dt = r*C*(1 - C/K) + delta*Y*C
      r*C*(1-C/K) : 로지스틱 증식항. 암세포는 초기(C≪K)에는 거의 지수적으로
                    (r*C) 증식하지만, 혈액공급·공간 등 자원의 한계 K(수용
                    용량, carrying capacity)에 가까워질수록 (1-C/K) 항이
                    0에 가까워지면서 증식이 자연히 둔화된다. 개체군생태학의
                    표준 로지스틱 성장모형과 수학적으로 동일한 구조다.
      delta*Y*C   : YAP/TAZ 신호가 암세포 증식을 추가로 가속시키는 항.
                    YAP는 세포증식·항세포사멸 유전자를 전사활성화하므로,
                    YAP 농도(Y)가 높을수록 이미 있는 암세포 수(C)에 비례해서
                    증식 속도가 더 빨라진다. 여기서도 Y와 C의 곱으로 쓰는
                    이유는 같은 질량작용 법칙이다 — YAP 신호가 암세포에
                    "작용"하려면 둘 다 존재해야 하기 때문이다.
    ---------------------------------------------------------------
    """
    Y, C = state
    dYdt = alpha * A - beta * M * Y - gamma * Y
    dCdt = r * C * (1 - C / K) + delta * Y * C
    return np.array([dYdt, dCdt])


def simulate(params, t_max=50, n_points=500, Y0=0.1, C0=1.0):
    """
    4차 룽게-쿠타(RK4)법으로 미분방정식을 수치적분하여
    시간에 따른 Y(t), C(t) 궤적을 반환한다.

    [왜 RK4를 직접 구현했는가]
    scipy.integrate.odeint를 쓰면 더 간단하지만, 배포 환경에서 scipy
    설치가 실패할 경우 앱 전체가 죽는 것을 막기 위해 외부 수치적분
    라이브러리 없이 numpy만으로 직접 구현한다.

    [RK4가 하는 일 — 각 스텝에서]
    현재 상태(Y, C)에서의 기울기(k1)를 구하고, 그 기울기로 절반만
    전진한 지점의 기울기(k2), 다시 k2로 절반 전진한 지점의 기울기(k3),
    k3로 한 스텝 전진한 지점의 기울기(k4)를 각각 구해 가중평균으로
    다음 상태를 계산한다. 오일러법(기울기 1번만 사용)보다 곡선이
    휘어지는 정도까지 반영되어(4차 정확도) 훨씬 정확하다.
    """
    t = np.linspace(0, t_max, n_points)
    dt = t[1] - t[0]
    args = (params["alpha"], params["beta"], params["gamma"],
            params["r"], params["K"], params["delta"],
            params["A"], params["M"])

    Y = np.zeros(n_points)
    C = np.zeros(n_points)
    Y[0], C[0] = Y0, C0

    for i in range(n_points - 1):
        state = np.array([Y[i], C[i]])
        k1 = yap_taz_odes(state, *args)
        k2 = yap_taz_odes(state + dt / 2 * k1, *args)
        k3 = yap_taz_odes(state + dt / 2 * k2, *args)
        k4 = yap_taz_odes(state + dt * k3, *args)

        new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # 암세포 수·YAP 농도가 음수로 내려가지 않도록(생물학적으로 불가능한 값 방지)
        Y[i + 1] = max(new_state[0], 0.0)
        C[i + 1] = max(new_state[1], 0.0)

    return t, Y, C


def steady_state_Y(alpha, A, beta, M, gamma):
    """
    YAP 정상상태(dY/dt=0) 농도의 해석해.

    0 = alpha*A - beta*M*Y* - gamma*Y*
    => Y*(beta*M + gamma) = alpha*A
    => Y* = alpha*A / (beta*M + gamma)

    M=0(운동을 전혀 하지 않을 때)이면 Y* = alpha*A/gamma로 최댓값을 갖고,
    M이 커질수록(운동을 많이 할수록) 분모가 커져서 Y*는 단조감소한다.
    beta, gamma > 0인 한 이 함수는 M에 대해 절대 증가할 수 없다는 점이
    탭 3의 디버깅 논리에서 핵심적으로 쓰인다.
    """
    return (alpha * A) / (beta * M + gamma)
