"""Generate a synthetic Indian credit-scoring dataset for fairness and bias auditing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_dataset(n: int = 50_000, output_path: str | Path | None = None, random_seed: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic lending dataset for Indian applicants.

    The dataset is intentionally synthetic and designed to support fairness analysis, SHAP
    interpretation, intersectional audits, and model-diagnostic exercises. It includes a
    demographic bias pattern across gender, age, city tier, income, and rural young women.
    """
    rng = np.random.default_rng(random_seed)

    states = np.array(["MH", "DL", "KA", "TN", "UP", "GJ", "WB", "RJ", "PB", "HR"])
    state_weights = np.array([0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.06, 0.04, 0.02])
    genders = np.array(["Female", "Male"])
    gender_probs = np.array([0.52, 0.48])
    city_tiers = np.array(["Tier 1", "Tier 2", "Tier 3"])

    gender = rng.choice(genders, size=n, p=gender_probs)
    rural_flag = np.where(
        gender == "Female",
        rng.random(n) < 0.42,
        rng.random(n) < 0.20,
    )

    city_prob_base = np.array([0.22, 0.48, 0.30])
    city_tier = np.array([
        rng.choice(city_tiers, p=city_prob_base) for _ in range(n)
    ])

    female_city_shift = np.where(
        gender == "Female",
        rng.random(n) < 0.58,
        rng.random(n) < 0.38,
    )
    city_tier = np.where(
        female_city_shift,
        np.where(rng.random(n) < 0.6, "Tier 3", np.where(rng.random(n) < 0.5, "Tier 2", "Tier 1")),
        city_tier,
    )
    city_tier = np.where(
        gender == "Male",
        np.where(rng.random(n) < 0.18, "Tier 1", city_tier),
        city_tier,
    )

    age = np.clip(
        np.where(
            gender == "Female",
            rng.normal(34, 11, size=n),
            rng.normal(37, 12, size=n),
        ),
        21,
        68,
    ).round().astype(int)
    age = np.where(young_rural := ((age < 30) & rural_flag & (gender == "Female")), np.clip(age + 1, 21, 68), age)

    employment_type = np.array([
        rng.choice(["Salaried", "Self-Employed", "Contract"], p=[0.56, 0.28, 0.16])
        for _ in range(n)
    ])
    education = np.array([
        rng.choice(["School", "Graduate", "Postgraduate"], p=[0.24, 0.60, 0.16])
        for _ in range(n)
    ])
    marital_status = np.array([
        rng.choice(["Single", "Married", "Separated"], p=[0.41, 0.50, 0.09])
        for _ in range(n)
    ])

    num_dependents = rng.integers(0, 6, size=n)
    existing_loans = rng.integers(0, 5, size=n)

    income_base = np.clip(rng.lognormal(mean=10.9, sigma=0.46, size=n), 15000, 900000)
    income = income_base.astype(float)
    income *= np.where(gender == "Female", 0.82, 1.00)
    income *= np.where(rural_flag, 0.70, 1.00)
    income *= np.where(city_tier == "Tier 3", 0.78, 1.00)
    income *= np.where(city_tier == "Tier 1", 1.18, 1.00)
    income = np.clip(income, 15000, 900000).round().astype(int)

    credit_bureau_score = np.clip(
        rng.normal(680, 90, size=n)
        + np.where(gender == "Female", -15, 0)
        + np.where((gender == "Female") & rural_flag, -25, 0)
        + np.where(city_tier == "Tier 3", -20, 0)
        + np.where(age < 25, -18, 0),
        300,
        850,
    ).round().astype(int)

    bank_relationship_years = np.clip(rng.normal(6.5, 4.2, size=n), 0, 20).round().astype(int)
    digital_transactions = np.clip(
        rng.normal(42000, 23000, size=n)
        + np.where(gender == "Male", 11000, 0)
        + np.where(city_tier == "Tier 1", 18000, 0)
        + np.where(rural_flag, -13000, 0),
        1500,
        600000,
    ).round().astype(int)

    mobile_os = rng.choice(["Android", "iOS"], size=n, p=[0.70, 0.30])
    loan_amount = np.clip(
        rng.lognormal(mean=10.7, sigma=0.55, size=n)
        * np.where(gender == "Female", 0.9, 1.12)
        * np.where(rural_flag, 0.88, 1.0),
        8000,
        800000,
    ).round().astype(int)

    loan_purpose = np.array([
        rng.choice(["Home", "Education", "Medical", "Travel", "Business", "Personal"], p=[0.17, 0.12, 0.18, 0.14, 0.22, 0.17])
        for _ in range(n)
    ])
    state = np.array([rng.choice(states, p=state_weights) for _ in range(n)])
    pincode = np.array([f"{rng.integers(110000, 999999):06d}" for _ in range(n)])
    employer_tier = np.array([
        rng.choice(["A", "B", "C"], p=[0.34, 0.41, 0.25])
        for _ in range(n)
    ])

    applicant_id = np.arange(1, n + 1)

    credit_signal = (
        0.38 * ((credit_bureau_score - 650) / 130)
        + 0.24 * ((income - 250000) / 180000)
        + 0.16 * ((bank_relationship_years - 6) / 8)
        + 0.12 * ((digital_transactions - 45000) / 40000)
        + 0.08 * ((education == "Graduate").astype(float) - 0.55)
        + 0.06 * ((city_tier == "Tier 1").astype(float) - 0.22)
        - 0.10 * ((gender == "Female").astype(float) - 0.52)
        - 0.09 * ((rural_flag).astype(float) - 0.30)
        - 0.08 * ((city_tier == "Tier 3").astype(float) - 0.30)
        - 0.15 * ((age < 30).astype(float) - 0.22)
    )

    raw_ai_score = 650 + 90 * credit_signal + rng.normal(0, 40, size=n)
    credit_ai_score = np.clip(raw_ai_score, 300, 900).round().astype(int)

    approval_threshold = 645
    approval_threshold += np.where(gender == "Female", -18, 10)
    approval_threshold += np.where(rural_flag, -14, 0)
    approval_threshold += np.where(city_tier == "Tier 3", -16, 0)
    approval_threshold += np.where((gender == "Female") & (age < 30), -12, 0)
    approval_threshold = np.round(approval_threshold).astype(float)
    credit_ai_decision = (credit_ai_score >= approval_threshold).astype(int)

    loan_to_income = loan_amount / np.maximum(income, 1)
    actual_default_logit = (
        -2.2
        + 0.014 * loan_to_income
        + 0.020 * np.maximum(existing_loans, 0)
        + 0.014 * np.maximum(num_dependents, 0)
        - 0.009 * credit_bureau_score / 100
        + 0.015 * (age < 25).astype(float)
        + 0.025 * (city_tier == "Tier 3").astype(float)
        + 0.020 * rural_flag.astype(float)
        + 0.012 * (gender == "Female").astype(float)
        + 0.035 * ((gender == "Female") & (age < 30) & rural_flag).astype(float)
    )
    actual_default_prob = 1 / (1 + np.exp(-actual_default_logit))
    actual_default_6m = (rng.random(n) < actual_default_prob).astype(int)

    # Ensure the model reflects discrimination against young rural women while leaving the overall
    # dataset realistic and measurable for fairness checks.
    actual_default_6m = np.where((gender == "Female") & rural_flag & (age < 30), np.where(rng.random(n) < 0.72, 1, actual_default_6m), actual_default_6m)
    credit_ai_decision = np.where((gender == "Female") & rural_flag & (age < 30), np.where(credit_ai_score >= 650, 1, 0), credit_ai_decision)
    credit_ai_decision = np.where((gender == "Male") & (credit_ai_score >= 650), 1, credit_ai_decision)

    # Keep backwards-compatible names used elsewhere in the project.
    credit_score = credit_ai_score.copy()
    model_score = credit_ai_score + rng.normal(0, 8, size=n)
    six_month_default = actual_default_6m.copy()
    credit_decision = credit_ai_decision.copy()

    df = pd.DataFrame(
        {
            "applicant_id": applicant_id,
            "age": age,
            "gender": gender,
            "state": state,
            "city_tier": city_tier,
            "rural_flag": rural_flag.astype(int),
            "pincode": pincode,
            "annual_income": income,
            "employment_type": employment_type,
            "employer_tier": employer_tier,
            "education": education,
            "marital_status": marital_status,
            "num_dependents": num_dependents,
            "existing_loans": existing_loans,
            "credit_bureau_score": credit_bureau_score,
            "bank_relationship_years": bank_relationship_years,
            "digital_transactions": digital_transactions,
            "mobile_os": mobile_os,
            "loan_amount": loan_amount,
            "loan_purpose": loan_purpose,
            "credit_score": credit_score,
            "model_score": model_score.round(2),
            "credit_ai_score": credit_ai_score,
            "credit_ai_decision": credit_ai_decision,
            "actual_default_6m": actual_default_6m,
            "six_month_default": six_month_default,
            "credit_decision": credit_decision,
        }
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        excel_path = output_path.with_suffix(".xlsx")
        df.to_excel(excel_path, index=False)

    return df


if __name__ == "__main__":
    generate_dataset(output_path="data/credit_data_synthetic.csv")
