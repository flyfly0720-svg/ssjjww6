
# -*- coding: utf-8 -*-
"""
Cox 비례위험회귀분석 - numpy 전용 구현 (lifelines 등 외부 통계 라이브러리 불필요)

[왜 자체 구현을 쓰는가]
Streamlit Cloud 배포 시 lifelines, scipy 같은 무거운 패키지는 설치가
느리거나 환경에 따라 실패하는 사례가 잦다. 이 앱은 배포 안정성을
최우선으로 두어, numpy와 pandas만으로 Cox 회귀의 핵심 계산(부분우도
최대화)을 직접 구현한다.

[Cox 부분우도(partial likelihood)란]
Cox 모형은 기저위험함수 h0(t)의 구체적 형태를 몰라도 회�귀계수 beta를
추정할 수 있는데, 이는 각 사건 발생 시점마다 "그 시점에 아직 사건이
일어나지 않고 남아있던 사람들(위험집합, risk set) 중에서 하필 이
사람에게 사건이 일어날 조건부 확률"을 곱해나가는 부분우도를 쓰기
때문이다.

사건이 발생한 시점 t_i에서 위험집합 R(t_i) = {j : time_j >= t_i}일 때,
부분우도의 로그는 다음과 같다.

L(beta) = sum_i [ x_i·beta - log( sum_{j in R(t_i)} exp(x_j·beta) ) ]

이 함수를 beta에 대해 최대화하는 것이 Cox 회귀의 목표이며,
동점(같은 시점에 여러 사건)이 있을 때는 Breslow 근사를 사용한다.

[왜 Newton-Raphson으로 푸는가]
L(beta)은 오목함수(concave)라서 유일한 최댓값을 갖는다. Newton-Raphson은
매 반복마다 1차 도함수(스코어함수, score)와 2차 도함수(헤시안, Hessian)를
이용해 접선의 근을 따라가며 최댓값에 빠르게(2차 수렴) 접근하는 방법으로,
로지스틱 회귀의 IRLS와 원리가 동일하다.
"""

import numpy as np


def _breslow_partial_likelihood_grad_hess(beta, X, time, event):
    """
    주어진 beta에서 로그부분우도의 값, 스코어(1차 도함수), 헤시안(2차 도함수)을
    한 번에 계산한다. 세 가지를 동시에 계산하는 이유는, 위험집합을 훑는
    반복문을 세 번 따로 돌리지 않고 한 번에 끝내 계산 비용을 줄이기 위함이다.

    각 단계가 왜 이렇게 되는지:
    ---------------------------------------------------------------
    1) risk_score = exp(X·beta) : 각 개인의 상대위험도(선형예측치의 지수).
       Cox 모형이 h(t|X) = h0(t)*exp(X·beta) 형태이므로, 부분우도 계산에는
       공통 인자 h0(t)가 약분되어 사라지고 exp(X·beta)만 남는다.

    2) 사건 발생 시점들을 오름차순으로 정렬한 뒤, 시점이 가장 늦은
       사람부터 위험집합에 누적해서 더해나간다(뒤에서부터 누적합).
       이렇게 하면 "이 시점에 아직 사건을 겪지 않고 남아있는 사람 전체"의
       합을 매번 새로 계산하지 않고 재사용할 수 있어 효율적이다.

    3) 각 사건 시점에서:
       - 로그우도 기여분 = x_i·beta - log(위험집합의 risk_score 합)
       - 스코어 기여분 = x_i - (위험집합의 x*risk_score 가중평균)
         : "실제 관측된 공변량"과 "위험집합에서 기대되는 평균 공변량"의
           차이. 이 차이가 0이 되는 지점이 최적해(beta_hat)이다.
       - 헤시안 기여분 = -(위험집합에서의 공변량 가중공분산)
         : 위험집합 내 공변량들이 얼마나 퍼져있는지(분산)에 따라
           결정되며, 오목함수이므로 항상 음의 정부호 행렬이 된다.
    ---------------------------------------------------------------
    """
    n, p = X.shape
    risk_score = np.exp(X @ beta)  # (n,)

    # 사건 시점 기준 내림차순 정렬: 시간이 늦은 사람부터 위험집합에 누적
    order = np.argsort(-time)
    time_sorted = time[order]
    event_sorted = event[order]
    X_sorted = X[order]
    rs_sorted = risk_score[order]

    # 누적합: cum_rs[i] = time_sorted[i] 이상인 모든 사람의 risk_score 합
    cum_rs = np.cumsum(rs_sorted)
    cum_rsX = np.cumsum(X_sorted * rs_sorted[:, None], axis=0)
    # 헤시안 계산용: 위험집합의 X'X 가중합 (p x p 행렬의 누적합)
    outer_terms = np.einsum('ni,nj->nij', X_sorted, X_sorted) * rs_sorted[:, None, None]
    cum_outer = np.cumsum(outer_terms, axis=0)

    loglik = 0.0
    score = np.zeros(p)
    hessian = np.zeros((p, p))

    event_idx = np.where(event_sorted == 1)[0]
    for i in event_idx:
        S0 = cum_rs[i]                       # 위험집합의 risk_score 합
        S1 = cum_rsX[i]                      # 위험집합의 X*risk_score 합
        S2 = cum_outer[i]                    # 위험집합의 X'X*risk_score 합

        loglik += X_sorted[i] @ beta - np.log(S0)
        mean_X = S1 / S0                     # 위험집합에서 기대되는 평균 공변량
        score += X_sorted[i] - mean_X
        # 공분산 = E[XX'] - E[X]E[X]'
        hessian -= (S2 / S0) - np.outer(mean_X, mean_X)

    return loglik, score, hessian


def fit_cox(X, time, event, max_iter=50, tol=1e-6):
    """
    Newton-Raphson 반복으로 Cox 회귀계수 beta를 추정한다.

    각 반복(iteration)이 하는 일:
      1) 현재 beta에서 로그우도, 스코어, 헤시안을 계산
      2) beta_new = beta - Hessian^{-1} @ score 로 갱신
         (score=0이 되는 지점, 즉 로그우도가 최대가 되는 지점을 향해 이동)
      3) beta가 거의 변하지 않으면(||변화량|| < tol) 수렴으로 보고 종료

    반환값: beta(회귀계수), se(표준오차, 헤시안 역행렬의 대각원소 제곱근),
            loglik(최종 로그부분우도)
    """
    n, p = X.shape
    beta = np.zeros(p)

    for _ in range(max_iter):
        loglik, score, hessian = _breslow_partial_likelihood_grad_hess(beta, X, time, event)
        try:
            step = np.linalg.solve(hessian, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, score, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    loglik, score, hessian = _breslow_partial_likelihood_grad_hess(beta, X, time, event)
    # 헤시안은 -Fisher 정보행렬이므로, 공분산행렬은 -Hessian의 역행렬
    cov = np.linalg.inv(-hessian)
    se = np.sqrt(np.diag(cov))

    return beta, se, loglik


def summary_table(beta, se, names):
    """
    회귀계수(beta), 표준오차(se)로부터 위험비(HR), 95% 신뢰구간, p-value를
    계산해 표 형태(딕셔너리 리스트)로 반환한다.

    - HR = exp(beta) : 해당 변수가 1단위 증가할 때 위험이 몇 배가 되는지
    - 95% CI = exp(beta ± 1.96*se) : beta의 정규근사 신뢰구간을 지수변환
      (1.96은 표준정규분포에서 양쪽 2.5%씩 자르는 z값)
    - p-value = 2*(1 - Phi(|z|)), z = beta/se : beta가 0이라는 귀무가설에서
      벗어난 정도를 표준정규분포 기준으로 환산. 정규분포의 누적분포함수는
      오차함수(erf)로 계산한다(외부 통계 라이브러리 없이 수학 공식만으로 계산).
    """
    from math import erf, sqrt

    def norm_cdf(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    rows = []
    for b, s, name in zip(beta, se, names):
        z = b / s if s > 0 else 0.0
        p = 2 * (1 - norm_cdf(abs(z)))
        hr = np.exp(b)
        ci_low = np.exp(b - 1.96 * s)
        ci_high = np.exp(b + 1.96 * s)
        rows.append({
            "변수": name, "beta": b, "se": s, "HR": hr,
            "CI_low": ci_low, "CI_high": ci_high, "p": p,
        })
    return rows
