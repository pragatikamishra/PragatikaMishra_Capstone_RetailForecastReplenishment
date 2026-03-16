from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
import importlib.util


BASE_DIR = Path(__file__).resolve().parent


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_step(title: str) -> None:
    print(f"\n--- {title} ---")


def load_module_from_path(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(module_name: str, file_path: Path):
    start = time.time()
    try:
        module = load_module_from_path(module_name, file_path)

        if not hasattr(module, "main"):
            raise AttributeError(f"{file_path.name} does not expose a main() function")

        module.main()
        duration = time.time() - start
        return True, duration, None

    except Exception as e:
        duration = time.time() - start
        error_text = f"FAILED: {file_path.name}\nReason: {e}\n\n{traceback.format_exc()}"
        return False, duration, error_text


def summarize_outputs():
    print_header("OUTPUT SUMMARY")

    expected_paths = [
        BASE_DIR / "data" / "fact_sales_store_sku_daily.csv",
        BASE_DIR / "data" / "fact_inventory_store_sku_daily.csv",
        BASE_DIR / "data" / "replenishment_inputs_store_sku.csv",
        BASE_DIR / "analysis" / "outputs" / "demand_trend_daily.csv",
        BASE_DIR / "analysis" / "outputs" / "weekly_pattern.csv",
        BASE_DIR / "analysis" / "outputs" / "promo_holiday_effect.csv",
        BASE_DIR / "analysis" / "outputs" / "top_skus_by_volume.csv",
        BASE_DIR / "analysis" / "outputs" / "top_skus_by_revenue.csv",
        BASE_DIR / "analysis" / "outputs" / "forecast_metrics_store_sku.csv",
        BASE_DIR / "analysis" / "outputs" / "forecast_detail_28d.csv",
        BASE_DIR / "analysis" / "outputs" / "forecast_metrics_overall.csv",
        BASE_DIR / "analysis" / "outputs" / "forecast_metrics_by_category.csv",
        BASE_DIR / "analysis" / "outputs" / "inventory_risk_segmented.csv",
        BASE_DIR / "analysis" / "outputs" / "top10_stockout_risk.csv",
        BASE_DIR / "analysis" / "outputs" / "top10_overstock_risk.csv",
        BASE_DIR / "analysis" / "outputs" / "replenishment_investigation_table.csv",
        BASE_DIR / "analysis" / "outputs" / "kpi_summary.csv",
        BASE_DIR / "analysis" / "outputs" / "replenishment_policy_store_sku.csv",
        BASE_DIR / "analysis" / "outputs" / "replenishment_policy_summary.csv",
        BASE_DIR / "analysis" / "outputs" / "replenishment_constraint_breakdown.csv",
        BASE_DIR / "analysis" / "outputs" / "replenishment_policy_by_category.csv",
        BASE_DIR / "analysis" / "outputs" / "recommended_po_list_next_cycle.csv",
        BASE_DIR / "analysis" / "outputs" / "impact_projection_store_sku_base.csv",
        BASE_DIR / "analysis" / "outputs" / "impact_projection_store_sku_scenarios.csv",
        BASE_DIR / "analysis" / "outputs" / "impact_projection_summary_by_scenario.csv",
        BASE_DIR / "analysis" / "outputs" / "impact_projection_by_category_base_case.csv",
        BASE_DIR / "analysis" / "outputs" / "top_impact_store_sku_base_case.csv",
        BASE_DIR / "analysis" / "outputs" / "impact_projection_assumptions.csv",
    ]

    found = 0
    missing = 0

    for path in expected_paths:
        if path.exists():
            print(f"[OK]      {path.relative_to(BASE_DIR)}")
            found += 1
        else:
            print(f"[MISSING] {path.relative_to(BASE_DIR)}")
            missing += 1

    print(f"\nFound   : {found}")
    print(f"Missing : {missing}")


def main() -> int:
    print_header("RETAIL FORECASTING & REPLENISHMENT PIPELINE RUNNER")
    print("Mode: source datasets loaded directly from URLs in etl/etl_pipeline.py")

    pipeline_steps = [
        ("Part B - ETL Pipeline", BASE_DIR / "etl" / "etl_pipeline.py", "etl_pipeline"),
        ("Part C - Analytics & Forecasting", BASE_DIR / "analysis" / "analysis.py", "analysis"),
        ("Part D - Replenishment Policy", BASE_DIR / "analysis" / "part_d_replenishment.py", "part_d_replenishment"),
        ("Part E - Impact Estimation", BASE_DIR / "analysis" / "part_e_impact_estimation.py", "part_e_impact_estimation"),
    ]

    results = []

    for title, file_path, module_name in pipeline_steps:
        print_step(title)
        ok, duration, error = run_script(module_name, file_path)

        results.append({
            "title": title,
            "ok": ok,
            "duration": duration,
        })

        if ok:
            print(f"SUCCESS in {duration:.2f}s")
        else:
            print(f"FAILED in {duration:.2f}s")
            print("\n" + error)
            print("\nStopping execution because this step is required for the next one.")
            break

    print_header("RUN SUMMARY")
    for result in results:
        status = "SUCCESS" if result["ok"] else "FAILED"
        print(f"{status:8} | {result['title']} | {result['duration']:.2f}s")

    all_ok = all(r["ok"] for r in results) and len(results) == len(pipeline_steps)

    if all_ok:
        summarize_outputs()
        print_header("NEXT MANUAL DELIVERABLES")
        print("1. analysis.ipynb narrative and charts")
        print("2. dashboard/retail_dashboard.xlsx")
        print("3. final_story/final_memo.pdf")
        print("4. README.md")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())