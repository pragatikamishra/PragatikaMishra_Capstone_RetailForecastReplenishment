from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Metrics
# =========================================================
def wape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom


def mape_safe(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


# =========================================================
# Load inputs
# =========================================================
def load_inputs():
    fact_sales = pd.read_csv(DATA_DIR / "fact_sales_store_sku_daily.csv", parse_dates=["date"])
    fact_inventory = pd.read_csv(DATA_DIR / "fact_inventory_store_sku_daily.csv", parse_dates=["date"])
    repl_inputs = pd.read_csv(DATA_DIR / "replenishment_inputs_store_sku.csv")

    fact_sales.columns = fact_sales.columns.str.strip().str.lower()
    fact_inventory.columns = fact_inventory.columns.str.strip().str.lower()
    repl_inputs.columns = repl_inputs.columns.str.strip().str.lower()

    return fact_sales, fact_inventory, repl_inputs


# =========================================================
# Part C1 — Demand understanding
# =========================================================
def demand_trend_outputs(sales: pd.DataFrame):
    daily_trend = (
        sales.groupby("date", as_index=False)
        .agg(
            units_sold=("units_sold", "sum"),
            true_demand_units=("true_demand_units", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_values("date")
    )
    daily_trend["units_sold_7d_ma"] = daily_trend["units_sold"].rolling(7, min_periods=1).mean()
    daily_trend["true_demand_7d_ma"] = daily_trend["true_demand_units"].rolling(7, min_periods=1).mean()
    daily_trend.to_csv(OUTPUT_DIR / "demand_trend_daily.csv", index=False)

    weekly_pattern = (
        sales.groupby("day_of_week", as_index=False)
        .agg(
            avg_units_sold=("units_sold", "mean"),
            avg_true_demand=("true_demand_units", "mean"),
            avg_revenue=("revenue", "mean"),
        )
    )

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekly_pattern["day_of_week"] = pd.Categorical(weekly_pattern["day_of_week"], dow_order, ordered=True)
    weekly_pattern = weekly_pattern.sort_values("day_of_week")
    weekly_pattern.to_csv(OUTPUT_DIR / "weekly_pattern.csv", index=False)

    promo_holiday_effect = (
        sales.groupby(["promo_flag", "holiday_flag"], as_index=False)
        .agg(
            avg_units_sold=("units_sold", "mean"),
            avg_true_demand=("true_demand_units", "mean"),
            avg_revenue=("revenue", "mean"),
            observations=("sku_id", "count"),
        )
    )
    promo_holiday_effect.to_csv(OUTPUT_DIR / "promo_holiday_effect.csv", index=False)

    top_skus_volume = (
        sales.groupby(["sku_id", "category"], as_index=False)
        .agg(total_units_sold=("units_sold", "sum"))
        .sort_values("total_units_sold", ascending=False)
        .head(20)
    )
    top_skus_volume.to_csv(OUTPUT_DIR / "top_skus_by_volume.csv", index=False)

    top_skus_revenue = (
        sales.groupby(["sku_id", "category"], as_index=False)
        .agg(total_revenue=("revenue", "sum"))
        .sort_values("total_revenue", ascending=False)
        .head(20)
    )
    top_skus_revenue.to_csv(OUTPUT_DIR / "top_skus_by_revenue.csv", index=False)


# =========================================================
# Part C2 — Forecasting
# =========================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "sku_id", "date"]).copy()

    df["dow_num"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    grouped = df.groupby(["store_id", "sku_id"])["true_demand_units"]

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)

    df["rolling_mean_7"] = (
        df.groupby(["store_id", "sku_id"])["true_demand_units"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )

    df["rolling_std_7"] = (
        df.groupby(["store_id", "sku_id"])["true_demand_units"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).std())
    )

    df["rolling_mean_28"] = (
        df.groupby(["store_id", "sku_id"])["true_demand_units"]
        .transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    )

    return df


def seasonal_naive_forecast(train_series: pd.Series, horizon: int = 28, season_len: int = 7) -> np.ndarray:
    train_series = train_series.reset_index(drop=True)
    if len(train_series) < season_len:
        return np.repeat(train_series.mean() if len(train_series) > 0 else 0, horizon)

    last_season = train_series.iloc[-season_len:].to_numpy()
    reps = int(np.ceil(horizon / season_len))
    return np.tile(last_season, reps)[:horizon]


def forecast_per_store_sku(sales: pd.DataFrame):
    df = add_time_features(sales)

    feature_cols = [
        "promo_flag", "holiday_flag", "dow_num", "month",
        "lag_1", "lag_7", "rolling_mean_7", "rolling_std_7", "rolling_mean_28"
    ]

    metrics_rows = []
    forecast_rows = []

    for (store_id, sku_id), grp in df.groupby(["store_id", "sku_id"]):
        grp = grp.sort_values("date").copy()
        if len(grp) < 84:
            continue

        train = grp.iloc[:-28].copy()
        test = grp.iloc[-28:].copy()

        y_train = train["true_demand_units"]
        y_test = test["true_demand_units"]

        baseline_pred = seasonal_naive_forecast(y_train, horizon=28, season_len=7)

        train_model = train.dropna(subset=feature_cols + ["true_demand_units"]).copy()
        test_model = test.dropna(subset=feature_cols + ["true_demand_units"]).copy()

        reg_pred = np.full(len(test), np.nan)

        if len(train_model) >= 30 and len(test_model) > 0:
            model = LinearRegression()
            model.fit(train_model[feature_cols], train_model["true_demand_units"])
            pred = model.predict(test_model[feature_cols])
            reg_pred[test.index.get_indexer(test_model.index)] = np.clip(pred, 0, None)

        baseline_mape = mape_safe(y_test, baseline_pred)
        baseline_wape = wape(y_test, baseline_pred)

        valid_reg = ~np.isnan(reg_pred)
        if valid_reg.any():
            regression_mape = mape_safe(y_test.iloc[valid_reg], reg_pred[valid_reg])
            regression_wape = wape(y_test.iloc[valid_reg], reg_pred[valid_reg])
        else:
            regression_mape = np.nan
            regression_wape = np.nan

        metrics_rows.append({
            "store_id": store_id,
            "sku_id": sku_id,
            "category": grp["category"].iloc[0],
            "baseline_mape": baseline_mape,
            "baseline_wape": baseline_wape,
            "regression_mape": regression_mape,
            "regression_wape": regression_wape,
        })

        for i, row in enumerate(test.itertuples(index=False)):
            forecast_rows.append({
                "date": row.date,
                "store_id": store_id,
                "sku_id": sku_id,
                "category": row.category,
                "actual_units_sold": row.units_sold,
                "actual_true_demand": row.true_demand_units,
                "baseline_forecast": baseline_pred[i],
                "regression_forecast": reg_pred[i],
            })

    metrics_df = pd.DataFrame(metrics_rows)
    forecast_df = pd.DataFrame(forecast_rows)

    return metrics_df, forecast_df


def summarize_forecast_performance(metrics_df: pd.DataFrame):
    overall = pd.DataFrame([{
        "baseline_mape_mean": metrics_df["baseline_mape"].mean(),
        "baseline_wape_mean": metrics_df["baseline_wape"].mean(),
        "regression_mape_mean": metrics_df["regression_mape"].mean(),
        "regression_wape_mean": metrics_df["regression_wape"].mean(),
        "better_model_by_wape": "regression"
        if metrics_df["regression_wape"].mean() < metrics_df["baseline_wape"].mean()
        else "baseline"
    }])

    by_category = (
        metrics_df.groupby("category", as_index=False)
        .agg(
            baseline_mape=("baseline_mape", "mean"),
            baseline_wape=("baseline_wape", "mean"),
            regression_mape=("regression_mape", "mean"),
            regression_wape=("regression_wape", "mean"),
        )
    )
    by_category["better_model_by_wape"] = np.where(
        by_category["regression_wape"] < by_category["baseline_wape"],
        "regression",
        "baseline"
    )

    overall.to_csv(OUTPUT_DIR / "forecast_metrics_overall.csv", index=False)
    by_category.to_csv(OUTPUT_DIR / "forecast_metrics_by_category.csv", index=False)

    return overall, by_category


# =========================================================
# Part C3 — Inventory risk segmentation
# =========================================================
def segment_inventory_risk(fact_inventory: pd.DataFrame, repl_inputs: pd.DataFrame, forecast_detail: pd.DataFrame):
    latest_inventory = (
        fact_inventory.sort_values("date")
        .groupby(["store_id", "sku_id"], as_index=False)
        .tail(1)
    )

    forecast_summary = (
        forecast_detail.groupby(["store_id", "sku_id", "category"], as_index=False)
        .agg(
            forecast_avg_daily_demand=("regression_forecast", "mean"),
            forecast_28d_units=("regression_forecast", "sum"),
        )
    )

    risk = latest_inventory.merge(repl_inputs, on=["store_id", "sku_id"], how="left", suffixes=("", "_repl"))
    risk = risk.merge(forecast_summary, on=["store_id", "sku_id"], how="left")

    if "category_x" in risk.columns and "category_y" in risk.columns:
        risk["category"] = risk["category_x"].fillna(risk["category_y"])
    elif "category_x" in risk.columns:
        risk["category"] = risk["category_x"]
    elif "category_y" in risk.columns:
        risk["category"] = risk["category_y"]

    risk["forecast_avg_daily_demand"] = risk["forecast_avg_daily_demand"].fillna(risk["avg_daily_demand"])
    risk["forecast_avg_daily_demand"] = risk["forecast_avg_daily_demand"].fillna(0)

    risk["projected_days_to_zero"] = np.where(
        risk["forecast_avg_daily_demand"] > 0,
        risk["on_hand_close"] / risk["forecast_avg_daily_demand"],
        np.inf
    )

    risk["doh_threshold"] = np.where(
        risk["shelf_life_days"].notna(),
        np.minimum(risk["shelf_life_days"] * 0.6, 30),
        21
    )

    risk["risk_flag"] = np.select(
        [
            risk["projected_days_to_zero"] < risk["lead_time_days"],
            risk["days_of_cover"] > risk["doh_threshold"],
        ],
        [
            "Stockout Risk",
            "Overstock Risk",
        ],
        default="Healthy Zone"
    )

    risk["recommended_action"] = np.select(
        [
            risk["risk_flag"] == "Stockout Risk",
            risk["risk_flag"] == "Overstock Risk",
        ],
        [
            "Replenish / expedite / transfer in",
            "Reduce reorder / transfer / promote out",
        ],
        default="Maintain policy"
    )

    stockout_top10 = (
        risk[risk["risk_flag"] == "Stockout Risk"]
        .sort_values(["projected_days_to_zero", "forecast_avg_daily_demand"], ascending=[True, False])
        .head(10)
    )

    overstock_top10 = (
        risk[risk["risk_flag"] == "Overstock Risk"]
        .sort_values(["days_of_cover"], ascending=False)
        .head(10)
    )

    risk.to_csv(OUTPUT_DIR / "inventory_risk_segmented.csv", index=False)
    stockout_top10.to_csv(OUTPUT_DIR / "top10_stockout_risk.csv", index=False)
    overstock_top10.to_csv(OUTPUT_DIR / "top10_overstock_risk.csv", index=False)

    return risk


# =========================================================
# Part C4 — Investigation table
# =========================================================
def build_replenishment_investigation_table(sales: pd.DataFrame, inventory: pd.DataFrame, risk_df: pd.DataFrame):
    cat_stockouts = (
        inventory.groupby("category", as_index=False)
        .agg(
            stockout_days=("stockout_flag", "sum"),
            total_days=("stockout_flag", "count"),
        )
    )
    cat_stockouts["stockout_rate"] = cat_stockouts["stockout_days"] / cat_stockouts["total_days"]

    top2 = cat_stockouts.sort_values("stockout_days", ascending=False).head(2)["category"].tolist()

    rows = []
    for category in top2:
        sales_cat = sales[sales["category"] == category]
        risk_cat = risk_df[(risk_df["category"] == category) & (risk_df["risk_flag"] == "Stockout Risk")]

        stores_contributing = (
            risk_cat.groupby("store_id", as_index=False)
            .agg(stockout_risk_skus=("sku_id", "nunique"))
            .sort_values("stockout_risk_skus", ascending=False)
            .head(5)
        )

        row_common = {
            "category": category,
            "stockout_rate": float(cat_stockouts.loc[cat_stockouts["category"] == category, "stockout_rate"].iloc[0]),
            "lost_sales_proxy_units": sales_cat["stockout_censored_units"].sum(),
            "stores_contributing_most": ", ".join(stores_contributing["store_id"].astype(str).tolist()),
            "avg_demand_volatility_stddev": sales_cat.groupby(["store_id", "sku_id"])["true_demand_units"].std().mean(),
        }

        rows.append({
            **row_common,
            "hypothesis": "Lead time is too long relative to demand velocity.",
            "follow_up_experiment_or_data": "Compare lead_time_days vs projected_days_to_zero by SKU-store.",
            "evidence_used": "inventory_risk_segmented.csv",
        })
        rows.append({
            **row_common,
            "hypothesis": "Promo and holiday demand spikes are not fully covered.",
            "follow_up_experiment_or_data": "Compare forecast error and stockout_censored_units on promo days.",
            "evidence_used": "promo_holiday_effect.csv",
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "replenishment_investigation_table.csv", index=False)


# =========================================================
# KPI summary
# =========================================================
def build_kpi_summary(sales: pd.DataFrame, inventory: pd.DataFrame, risk_df: pd.DataFrame, forecast_overall: pd.DataFrame):
    latest_inventory = (
        inventory.sort_values("date")
        .groupby(["store_id", "sku_id"], as_index=False)
        .tail(1)
    )

    inventory_value_proxy = (latest_inventory["on_hand_close"] * latest_inventory["cost"]).sum()

    true_demand_total = sales["true_demand_units"].sum()
    censored_total = sales["stockout_censored_units"].sum()

    fill_rate_proxy = np.nan
    if true_demand_total > 0:
        fill_rate_proxy = 1 - (censored_total / true_demand_total)

    kpis = pd.DataFrame([{
        "total_units_sold": sales["units_sold"].sum(),
        "total_true_demand_units": true_demand_total,
        "stockout_censored_units": censored_total,
        "total_revenue": sales["revenue"].sum(),
        "total_margin_proxy": sales["margin_proxy"].sum(),
        "stockout_rate": inventory["stockout_flag"].mean(),
        "fill_rate_proxy": fill_rate_proxy,
        "avg_days_of_inventory_on_hand": inventory["days_of_cover"].replace([np.inf, -np.inf], np.nan).mean(),
        "overstock_rate": (risk_df["risk_flag"] == "Overstock Risk").mean(),
        "inventory_value_proxy": inventory_value_proxy,
        "holding_cost_proxy": inventory_value_proxy * 0.02,
        "lost_sales_proxy_units": censored_total,
        "baseline_wape_mean": forecast_overall["baseline_wape_mean"].iloc[0],
        "regression_wape_mean": forecast_overall["regression_wape_mean"].iloc[0],
    }])

    kpis.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)


# =========================================================
# Main
# =========================================================
def main():
    print("Loading curated datasets...")
    fact_sales, fact_inventory, repl_inputs = load_inputs()

    print("Running demand understanding...")
    demand_trend_outputs(fact_sales)

    print("Running forecasting...")
    forecast_metrics, forecast_detail = forecast_per_store_sku(fact_sales)
    forecast_metrics.to_csv(OUTPUT_DIR / "forecast_metrics_store_sku.csv", index=False)
    forecast_detail.to_csv(OUTPUT_DIR / "forecast_detail_28d.csv", index=False)

    print("Summarizing forecast performance...")
    forecast_overall, _ = summarize_forecast_performance(forecast_metrics)

    print("Running inventory risk segmentation...")
    risk_df = segment_inventory_risk(fact_inventory, repl_inputs, forecast_detail)

    print("Building investigation table...")
    build_replenishment_investigation_table(fact_sales, fact_inventory, risk_df)

    print("Building KPI summary...")
    build_kpi_summary(fact_sales, fact_inventory, risk_df, forecast_overall)

    print(f"Done. Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()