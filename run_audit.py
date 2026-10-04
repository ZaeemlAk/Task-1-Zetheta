"""Generate a synthetic dataset and compute the initial fairness audit summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from src.data_quality import describe_dataset
from src.eda import plot_approval_rate_by_group
from src.fairness_metrics import approval_rate, demographic_parity_ratio
from src.generate_dataset import generate_dataset
from src.intersectional_analysis import intersectional_approval_rates
from src.proxy_detection import find_proxy_candidates


def main() -> None:
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
path = data_dir / "credit_data_synthetic.csv"
df = generate_dataset(output_path=path)

quality = describe_dataset(df)
print("Dataset rows:", quality["n_records"])
print("Dataset columns:", quality["n_columns"])
print("Duplicate rows:", quality["duplicate_rows"])

approval = approval_rate(df, "gender", "credit_decision")
print("\nApproval by gender:")
print(approval.to_string(index=False))

ratio = demographic_parity_ratio(df, "gender", "credit_decision", "Male")
print("\nDemographic parity ratio (relative to Male):", round(ratio, 4))

intersection = intersectional_approval_rates(df, ["gender", "city_tier"], "credit_decision")
print("\nIntersectional approval rates (top 10):")
print(intersection.sort_values("approval_rate").head(10).to_string(index=False))

proxy_candidates = find_proxy_candidates(df, ["gender"], threshold=0.03)
print("\nProxy candidates for gender:")
print(proxy_candidates)

summary = {
"rows": int(df.shape[0]),
"columns": int(df.shape[1]),
"approval_rate_overall": round(float(df["credit_decision"].mean()), 4),
"default_rate_among_approved": round(float(df.loc[df["credit_decision"] == 1, "six_month_default"].mean()), 4),
"approval_rate_female": round(float(df.loc[df["gender"] == "Female", "credit_decision"].mean()), 4),
"approval_rate_male": round(float(df.loc[df["gender"] == "Male", "credit_decision"].mean()), 4),
"demographic_parity_ratio_vs_male": round(float(ratio), 4),
}

    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    chart = plot_approval_rate_by_group(df, "gender", "credit_decision", "Approval rate by gender")
    chart.savefig(report_dir / "approval_rate_by_gender.png", dpi=200, bbox_inches="tight")

    text_report = "\n".join(
        [
            "Synthetic credit audit summary",
            "===========================",
            f"Rows: {summary['rows']}",
            f"Columns: {summary['columns']}",
            f"Overall approval rate: {summary['approval_rate_overall']}",
            f"Female approval rate: {summary['approval_rate_female']}",
            f"Male approval rate: {summary['approval_rate_male']}",
            f"Demographic parity ratio (vs Male): {summary['demographic_parity_ratio_vs_male']}",
            f"Default rate among approved applicants: {summary['default_rate_among_approved']}",
            "",
            "Interpretation:",
            "- The synthetic dataset shows a measurable approval-rate gap by gender.",
            "- The parity ratio is computed against the Male reference group to demonstrate a simple fairness check.",
            "- The model should be treated as a scaffolded audit example, not as a real-world financial decision system.",
        ]
    )
    (report_dir / "fairness_summary.txt").write_text(text_report, encoding="utf-8")

    markdown = "\n".join(
        [
            "# Synthetic Credit Bias Audit Summary",
            "",
            "This report is based on a synthetic dataset generated from the project brief and is intended for audit scaffolding and demonstration purposes.",
            "",
            "## Dataset overview",
            f"- Rows: {summary['rows']}",
            f"- Columns: {summary['columns']}",
            f"- Overall approval rate: {summary['approval_rate_overall']}",
            f"- Female approval rate: {summary['approval_rate_female']}",
            f"- Male approval rate: {summary['approval_rate_male']}",
            f"- Demographic parity ratio (vs Male): {summary['demographic_parity_ratio_vs_male']}",
            f"- Default rate among approved applicants: {summary['default_rate_among_approved']}",
            "",
            "## Audit observation",
            "The synthetic decision process produces a non-trivial approval-rate gap between female and male applicants. This is intended to demonstrate how fairness analysis is structured in practice and should not be interpreted as evidence about a real lending portfolio.",
            "",
            "## Proxy candidate check",
            f"`gender` proxy candidates: {proxy_candidates.get('gender', [])}",
            "",
            "![Approval rate by gender](approval_rate_by_gender.png)",
        ]
    )
    (report_dir / "fairness_summary.md").write_text(markdown, encoding="utf-8")

    print(f"\nSummary saved to {report_dir / 'fairness_summary.txt'}")
    print(f"Markdown summary saved to {report_dir / 'fairness_summary.md'}")


if __name__ == "__main__":
main()
