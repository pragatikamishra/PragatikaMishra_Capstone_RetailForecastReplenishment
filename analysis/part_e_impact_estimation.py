from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_inputs():
    fact_inventory = pd.read_csv(DATA_DIR / "fact_inventory_store_sku_daily.csv", parse_dates=["date"])
    fact_sales = pd.read_csv(DATA_DIR / "fact_sales_store_sku_daily.csv", parse_dates=["date"])
    repl_inputs = pd.read_csv(DATA_DIR / "replenishment_inputs_store_sku.csv")

    forecast_detail = pd.read_csv(OUTPUT_DIR / "forecast_detail_28d.csv", parse_dates=["date"])
    policy_df = pd.read_csv(OUTPUT_DIR / "replenishment_policy_store_sku.csv")

    return fact_inventory, fact_sales, repl_inputs, forecast_detail, policy_df


def build_base_projection_table(fact_inventory, fact_sales, repl_inputs, forecast_detail, policy_df):
    latest_inventory = (
        fact_inventory.sort_values("date")
        .groupby(["store_id", "sku_id"], as_index=False)
        .tail(1)[[
            "store_id", "sku_id", "date", "region", "city_tier", "store_size",
            "category", "price", "cost", "on_hand_close", "days_of_cover"
        ]]
        .rename(columns={"date": "inventory_date"})
    )

    hist_sales = (
        fact_sales.groupby(["store_id", "sku_id"], as_index=False)
        .agg(
            avg_daily_sales_hist=("true_demand_units", "mean"),
            demand_std_hist=("true_demand_units", "std"),
            stockout_censored_units_hist=("stockout_censored_units", "sum"),
        )
    )

    fc = forecast_detail.copy()
    fc["best_forecast"] = np.where(
        fc["regression_forecast"].notna(),
        fc["regression_forecast"],
        fc["baseline_forecast"]
    )

    fc_summary = (
        fc.groupby(["store_id", "sku_id"], as_index=False)
        .agg(
            forecast_units_28d=("best_forecast", "sum"),
            forecast_avg_daily=("best_forecast", "mean"),
        )
    )

    policy_cols = [
        "store_id", "sku_id", "recommended_order_qty", "should_reorder",
        "recommended_order_cost_value", "recommended_action"
    ]

    df = latest_inventory.merge(hist_sales, on=["store_id", "sku_id"], how="left")
    df = df.merge(repl_inputs[["store_id", "sku_id", "avg_daily_demand", "demand_std_dev", "lead_time_days"]], on=["store_id", "sku_id"], how="left")
    df = df.merge(fc_summary, on=["store_id", "sku_id"], how="left")
    df = df.merge(policy_df[policy_cols], on=["store_id", "sku_id"], how="left")

    df["daily_demand_for_projection"] = df["forecast_avg_daily"].fillna(df["avg_daily_demand"]).fillna(df["avg_daily_sales_hist"]).fillna(0)
    df["forecast_units_30d"] = df["daily_demand_for_projection"] * 30

    df["baseline_ending_inventory"] = df["on_hand_close"] - df["forecast_units_30d"]
    df["proposed_available_inventory"] = df["on_hand_close"] + df["recommended_order_qty"].fillna(0)
    df["proposed_ending_inventory"] = df["proposed_available_inventory"] - df["forecast_units_30d"]

    df["baseline_unmet_units"] = np.maximum(-df["baseline_ending_inventory"], 0)
    df["proposed_unmet_units"] = np.maximum(-df["proposed_ending_inventory"], 0)

    df["lost_sales_avoided_units"] = np.maximum(df["baseline_unmet_units"] - df["proposed_unmet_units"], 0)
    df["lost_sales_avoided_value"] = df["lost_sales_avoided_units"] * df["price"]

    df["baseline_stockout_days_proxy"] = np.where(
        df["daily_demand_for_projection"] > 0,
        np.ceil(df["baseline_unmet_units"] / df["daily_demand_for_projection"]),
        0
    )
    df["proposed_stockout_days_proxy"] = np.where(
        df["daily_demand_for_projection"] > 0,
        np.ceil(df["proposed_unmet_units"] / df["daily_demand_for_projection"]),
        0
    )
    df["stockout_days_reduced_proxy"] = np.maximum(
        df["baseline_stockout_days_proxy"] - df["proposed_stockout_days_proxy"],
        0
    )

    df["baseline_inventory_value_end"] = np.maximum(df["baseline_ending_inventory"], 0) * df["cost"]
    df["proposed_inventory_value_end"] = np.maximum(df["proposed_ending_inventory"], 0) * df["cost"]
    df["inventory_value_change"] = df["proposed_inventory_value_end"] - df["baseline_inventory_value_end"]

    monthly_holding_rate = 0.02
    df["baseline_holding_cost_proxy"] = df["baseline_inventory_value_end"] * monthly_holding_rate
    df["proposed_holding_cost_proxy"] = df["proposed_inventory_value_end"] * monthly_holding_rate
    df["holding_cost_change"] = df["proposed_holding_cost_proxy"] - df["baseline_holding_cost_proxy"]

    df["baseline_fill_rate_proxy"] = np.where(
        df["forecast_units_30d"] > 0,
        1 - (df["baseline_unmet_units"] / df["forecast_units_30d"]),
        1
    )
    df["proposed_fill_rate_proxy"] = np.where(
        df["forecast_units_30d"] > 0,
        1 - (df["proposed_unmet_units"] / df["forecast_units_30d"]),
        1
    )
    df["fill_rate_improvement"] = df["proposed_fill_rate_proxy"] - df["baseline_fill_rate_proxy"]

    return df


def apply_scenario(df: pd.DataFrame, scenario_name: str, demand_multiplier: float, lead_time_multiplier: float):
    out = df.copy()
    out["scenario"] = scenario_name

    out["scenario_daily_demand"] = out["daily_demand_for_projection"] * demand_multiplier
    out["scenario_units_30d"] = out["scenario_daily_demand"] * 30
    out["effective_lead_time_days"] = out["lead_time_days"].fillna(7) * lead_time_multiplier

    out["po_realization_factor"] = np.clip(
        1 - ((out["effective_lead_time_days"] - out["lead_time_days"].fillna(7)) / 30),
        0.5,
        1.05
    )

    out["scenario_effective_po_qty"] = out["recommended_order_qty"].fillna(0) * out["po_realization_factor"]

    out["scenario_baseline_ending_inventory"] = out["on_hand_close"] - out["scenario_units_30d"]
    out["scenario_proposed_ending_inventory"] = (out["on_hand_close"] + out["scenario_effective_po_qty"]) - out["scenario_units_30d"]

    out["scenario_baseline_unmet_units"] = np.maximum(-out["scenario_baseline_ending_inventory"], 0)
    out["scenario_proposed_unmet_units"] = np.maximum(-out["scenario_proposed_ending_inventory"], 0)

    out["scenario_lost_sales_avoided_units"] = np.maximum(
        out["scenario_baseline_unmet_units"] - out["scenario_proposed_unmet_units"],
        0
    )
    out["scenario_lost_sales_avoided_value"] = out["scenario_lost_sales_avoided_units"] * out["price"]

    out["scenario_baseline_stockout_days"] = np.where(
        out["scenario_daily_demand"] > 0,
        np.ceil(out["scenario_baseline_unmet_units"] / out["scenario_daily_demand"]),
        0
    )
    out["scenario_proposed_stockout_days"] = np.where(
        out["scenario_daily_demand"] > 0,
        np.ceil(out["scenario_proposed_unmet_units"] / out["scenario_daily_demand"]),
        0
    )
    out["scenario_stockout_days_reduced"] = np.maximum(
        out["scenario_baseline_stockout_days"] - out["scenario_proposed_stockout_days"],
        0
    )

    out["scenario_baseline_inventory_value_end"] = np.maximum(out["scenario_baseline_ending_inventory"], 0) * out["cost"]
    out["scenario_proposed_inventory_value_end"] = np.maximum(out["scenario_proposed_ending_inventory"], 0) * out["cost"]
    out["scenario_inventory_value_change"] = out["scenario_proposed_inventory_value_end"] - out["scenario_baseline_inventory_value_end"]
    out["scenario_holding_cost_change"] = out["scenario_inventory_value_change"] * 0.02

    out["scenario_baseline_fill_rate"] = np.where(
        out["scenario_units_30d"] > 0,
        1 - (out["scenario_baseline_unmet_units"] / out["scenario_units_30d"]),
        1
    )
    out["scenario_proposed_fill_rate"] = np.where(
        out["scenario_units_30d"] > 0,
        1 - (out["scenario_proposed_unmet_units"] / out["scenario_units_30d"]),
        1
    )
    out["scenario_fill_rate_improvement"] = out["scenario_proposed_fill_rate"] - out["scenario_baseline_fill_rate"]

    return out


def build_scenarios(base_df: pd.DataFrame):
    best = apply_scenario(base_df, "Best Case", demand_multiplier=0.95, lead_time_multiplier=0.90)
    base = apply_scenario(base_df, "Base Case", demand_multiplier=1.00, lead_time_multiplier=1.00)
    worst = apply_scenario(base_df, "Worst Case", demand_multiplier=1.10, lead_time_multiplier=1.20)
    return pd.concat([best, base, worst], ignore_index=True)


def summarize_scenarios(scenarios_df: pd.DataFrame):
    return (
        scenarios_df.groupby("scenario", as_index=False)
        .agg(
            sku_store_rows=("sku_id", "count"),
            total_baseline_unmet_units=("scenario_baseline_unmet_units", "sum"),
            total_proposed_unmet_units=("scenario_proposed_unmet_units", "sum"),
            lost_sales_avoided_units=("scenario_lost_sales_avoided_units", "sum"),
            lost_sales_avoided_value=("scenario_lost_sales_avoided_value", "sum"),
            stockout_days_reduced_proxy=("scenario_stockout_days_reduced", "sum"),
            inventory_value_change=("scenario_inventory_value_change", "sum"),
            holding_cost_change=("scenario_holding_cost_change", "sum"),
            avg_baseline_fill_rate=("scenario_baseline_fill_rate", "mean"),
            avg_proposed_fill_rate=("scenario_proposed_fill_rate", "mean"),
            avg_fill_rate_improvement=("scenario_fill_rate_improvement", "mean"),
        )
    )


def category_impact_summary(scenarios_df: pd.DataFrame, scenario_name: str = "Base Case"):
    df = scenarios_df[scenarios_df["scenario"] == scenario_name].copy()
    return (
        df.groupby("category", as_index=False)
        .agg(
            lost_sales_avoided_units=("scenario_lost_sales_avoided_units", "sum"),
            lost_sales_avoided_value=("scenario_lost_sales_avoided_value", "sum"),
            stockout_days_reduced_proxy=("scenario_stockout_days_reduced", "sum"),
            inventory_value_change=("scenario_inventory_value_change", "sum"),
            holding_cost_change=("scenario_holding_cost_change", "sum"),
        )
        .sort_values("lost_sales_avoided_value", ascending=False)
    )


def top_impact_skus(scenarios_df: pd.DataFrame, scenario_name: str = "Base Case", top_n: int = 20):
    df = scenarios_df[scenarios_df["scenario"] == scenario_name].copy()
    return (
        df.sort_values(
            ["scenario_lost_sales_avoided_value", "scenario_stockout_days_reduced"],
            ascending=[False, False]
        )
        .loc[:, [
            "store_id", "sku_id", "category", "region", "city_tier", "store_size",
            "on_hand_close", "recommended_order_qty",
            "scenario_lost_sales_avoided_units",
            "scenario_lost_sales_avoided_value",
            "scenario_stockout_days_reduced",
            "scenario_inventory_value_change",
            "scenario_holding_cost_change",
            "recommended_action"
        ]]
        .head(top_n)
    )


def build_assumptions_table():
    return pd.DataFrame([
        {"scenario": "Best Case", "assumption_type": "Demand", "assumption_value": "-5%", "explanation": "Demand slightly below forecast."},
        {"scenario": "Best Case", "assumption_type": "Lead Time", "assumption_value": "-10%", "explanation": "Orders land slightly faster."},
        {"scenario": "Base Case", "assumption_type": "Demand", "assumption_value": "0%", "explanation": "Demand follows forecast."},
        {"scenario": "Base Case", "assumption_type": "Lead Time", "assumption_value": "0%", "explanation": "Lead time stays normal."},
        {"scenario": "Worst Case", "assumption_type": "Demand", "assumption_value": "+10%", "explanation": "Demand exceeds forecast."},
        {"scenario": "Worst Case", "assumption_type": "Lead Time", "assumption_value": "+20%", "explanation": "Orders land later."},
        {"scenario": "All", "assumption_type": "Holding Cost", "assumption_value": "2% monthly", "explanation": "Holding cost proxy = ending inventory value × 2%."},
        {"scenario": "All", "assumption_type": "Lost Sales Proxy", "assumption_value": "Unmet units × price", "explanation": "Unmet units are treated as lost sales."},
    ])


def main():
    print("Loading inputs...")
    fact_inventory, fact_sales, repl_inputs, forecast_detail, policy_df = load_inputs()

    print("Building base projection...")
    base_df = build_base_projection_table(
        fact_inventory, fact_sales, repl_inputs, forecast_detail, policy_df
    )

    print("Running scenarios...")
    scenarios_df = build_scenarios(base_df)

    print("Building summaries...")
    scenario_summary = summarize_scenarios(scenarios_df)
    category_summary = category_impact_summary(scenarios_df, scenario_name="Base Case")
    top_skus = top_impact_skus(scenarios_df, scenario_name="Base Case", top_n=20)
    assumptions = build_assumptions_table()

    base_df.to_csv(OUTPUT_DIR / "impact_projection_store_sku_base.csv", index=False)
    scenarios_df.to_csv(OUTPUT_DIR / "impact_projection_store_sku_scenarios.csv", index=False)
    scenario_summary.to_csv(OUTPUT_DIR / "impact_projection_summary_by_scenario.csv", index=False)
    category_summary.to_csv(OUTPUT_DIR / "impact_projection_by_category_base_case.csv", index=False)
    top_skus.to_csv(OUTPUT_DIR / "top_impact_store_sku_base_case.csv", index=False)
    assumptions.to_csv(OUTPUT_DIR / "impact_projection_assumptions.csv", index=False)

    print(f"Done. Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()