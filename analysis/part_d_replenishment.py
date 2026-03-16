from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def z_value_from_service_level(service_level: float) -> float:
    lookup = {
        0.90: 1.28,
        0.91: 1.34,
        0.92: 1.41,
        0.93: 1.48,
        0.94: 1.55,
        0.95: 1.65,
        0.96: 1.75,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33,
    }
    return lookup.get(round(float(service_level), 2), 1.65)


def choose_target_days_of_cover(category: str) -> int:
    mapping = {
        "Dairy": 5,
        "Fresh": 4,
        "Produce": 4,
        "Bakery": 3,
        "Frozen": 14,
        "Beverages": 18,
        "Snacks": 21,
        "Staples": 28,
        "Personal Care": 30,
    }
    return mapping.get(str(category).strip().title(), 14)


def choose_max_storage_by_store_size(store_size: str) -> int:
    mapping = {
        "small": 500,
        "medium": 1200,
        "large": 2500,
    }
    return mapping.get(str(store_size).strip().lower(), 1000)


def load_inputs():
    repl_inputs = pd.read_csv(DATA_DIR / "replenishment_inputs_store_sku.csv")
    fact_inventory = pd.read_csv(DATA_DIR / "fact_inventory_store_sku_daily.csv", parse_dates=["date"])
    return repl_inputs, fact_inventory


def build_replenishment_policy(repl_inputs: pd.DataFrame, fact_inventory: pd.DataFrame) -> pd.DataFrame:
    latest_inventory = (
        fact_inventory.sort_values("date")
        .groupby(["store_id", "sku_id"], as_index=False)
        .tail(1)[["store_id", "sku_id", "date", "on_hand_close"]]
        .rename(columns={"date": "inventory_date", "on_hand_close": "current_on_hand"})
    )

    df = repl_inputs.merge(latest_inventory, on=["store_id", "sku_id"], how="left")

    df["current_on_hand"] = df["current_on_hand"].fillna(df["on_hand_units"]).fillna(0)
    df["service_level_target"] = pd.to_numeric(df["service_level_target"], errors="coerce").fillna(0.95)
    df["avg_daily_demand"] = pd.to_numeric(df["avg_daily_demand"], errors="coerce").fillna(0)
    df["demand_std_dev"] = pd.to_numeric(df["demand_std_dev"], errors="coerce").fillna(0)
    df["lead_time_days"] = pd.to_numeric(df["lead_time_days"], errors="coerce").fillna(7)
    df["moq_units"] = pd.to_numeric(df["moq_units"], errors="coerce").fillna(1).clip(lower=1)

    df["z_value"] = df["service_level_target"].apply(z_value_from_service_level)

    df["safety_stock_calc"] = (
        df["z_value"] * df["demand_std_dev"] * np.sqrt(df["lead_time_days"])
    )

    df["reorder_point_calc"] = (
        df["avg_daily_demand"] * df["lead_time_days"] + df["safety_stock_calc"]
    )

    df["target_days_of_cover"] = df["category"].apply(choose_target_days_of_cover)

    df["order_up_to_level"] = (
        df["avg_daily_demand"] * df["target_days_of_cover"] + df["safety_stock_calc"]
    )

    df["recommended_order_qty_raw"] = np.maximum(
        df["order_up_to_level"] - df["current_on_hand"],
        0
    )

    df["recommended_order_qty_after_moq"] = np.where(
        (df["recommended_order_qty_raw"] > 0) & (df["recommended_order_qty_raw"] < df["moq_units"]),
        df["moq_units"],
        df["recommended_order_qty_raw"]
    )

    df["max_units_by_shelf_life"] = np.where(
        df["shelf_life_days"].notna() & (df["avg_daily_demand"] > 0),
        df["avg_daily_demand"] * (df["shelf_life_days"] * 0.60),
        np.inf
    )

    df["recommended_order_qty_after_shelf_life"] = np.minimum(
        df["recommended_order_qty_after_moq"],
        df["max_units_by_shelf_life"]
    )

    df["max_storage_units"] = df["store_size"].apply(choose_max_storage_by_store_size)
    df["remaining_storage_capacity"] = np.maximum(df["max_storage_units"] - df["current_on_hand"], 0)

    df["recommended_order_qty_final"] = np.minimum(
        df["recommended_order_qty_after_shelf_life"],
        df["remaining_storage_capacity"]
    )

    df["should_reorder"] = df["current_on_hand"] <= df["reorder_point_calc"]

    df["recommended_order_qty"] = np.where(
        df["should_reorder"],
        np.ceil(df["recommended_order_qty_final"]),
        0
    )

    df["days_of_cover_current"] = np.where(
        df["avg_daily_demand"] > 0,
        df["current_on_hand"] / df["avg_daily_demand"],
        np.inf
    )

    df["recommended_action"] = np.select(
        [
            (df["should_reorder"]) & (df["recommended_order_qty"] > 0),
            (df["should_reorder"]) & (df["recommended_order_qty"] == 0),
            (~df["should_reorder"]),
        ],
        [
            "Raise PO",
            "Review constraint block",
            "No order needed",
        ],
        default="Review"
    )

    notes = []
    for row in df.itertuples(index=False):
        row_notes = []
        if row.recommended_order_qty_raw > 0 and row.recommended_order_qty_raw < row.moq_units:
            row_notes.append("MOQ applied")
        if row.recommended_order_qty_after_moq > row.recommended_order_qty_after_shelf_life:
            row_notes.append("Shelf-life cap applied")
        if row.recommended_order_qty_after_shelf_life > row.recommended_order_qty_final:
            row_notes.append("Storage cap applied")
        if not row.should_reorder:
            row_notes.append("Below reorder trigger not met")
        if not row_notes:
            row_notes.append("No constraint triggered")
        notes.append("; ".join(row_notes))

    df["constraint_notes"] = notes
    df["recommended_order_cost_value"] = df["recommended_order_qty"] * df["cost"]
    df["recommended_order_sales_value"] = df["recommended_order_qty"] * df["price"]

    final_cols = [
        "store_id",
        "sku_id",
        "inventory_date",
        "region",
        "city_tier",
        "store_size",
        "category",
        "price",
        "cost",
        "avg_daily_demand",
        "demand_std_dev",
        "lead_time_days",
        "service_level_target",
        "z_value",
        "safety_stock_calc",
        "reorder_point_calc",
        "target_days_of_cover",
        "order_up_to_level",
        "current_on_hand",
        "days_of_cover_current",
        "moq_units",
        "shelf_life_days",
        "max_storage_units",
        "remaining_storage_capacity",
        "should_reorder",
        "recommended_order_qty_raw",
        "recommended_order_qty",
        "recommended_action",
        "constraint_notes",
        "recommended_order_cost_value",
        "recommended_order_sales_value",
    ]

    return df[final_cols].sort_values(["store_id", "sku_id"]).reset_index(drop=True)


def build_replenishment_summaries(policy_df: pd.DataFrame):
    po_list = policy_df[(policy_df["should_reorder"]) & (policy_df["recommended_order_qty"] > 0)].copy()
    top_po_list = po_list.sort_values(
        ["recommended_order_cost_value", "recommended_order_qty"],
        ascending=[False, False]
    ).head(50)

    summary = pd.DataFrame([{
        "total_store_sku_rows": len(policy_df),
        "rows_triggering_reorder": int(policy_df["should_reorder"].sum()),
        "rows_with_positive_recommended_order": int((policy_df["recommended_order_qty"] > 0).sum()),
        "total_recommended_units": float(policy_df["recommended_order_qty"].sum()),
        "total_recommended_cost_value": float(policy_df["recommended_order_cost_value"].sum(skipna=True)),
        "avg_service_level_target": float(policy_df["service_level_target"].mean()),
        "avg_safety_stock": float(policy_df["safety_stock_calc"].mean()),
        "avg_reorder_point": float(policy_df["reorder_point_calc"].mean()),
    }])

    constraint_breakdown = pd.DataFrame({
        "constraint_notes": policy_df["constraint_notes"].value_counts().index,
        "row_count": policy_df["constraint_notes"].value_counts().values,
    })

    category_summary = (
        policy_df.groupby("category", as_index=False)
        .agg(
            sku_store_count=("sku_id", "count"),
            reorder_rows=("should_reorder", "sum"),
            total_recommended_units=("recommended_order_qty", "sum"),
            total_recommended_cost_value=("recommended_order_cost_value", "sum"),
            avg_safety_stock=("safety_stock_calc", "mean"),
            avg_reorder_point=("reorder_point_calc", "mean"),
        )
        .sort_values("total_recommended_cost_value", ascending=False)
    )

    return summary, constraint_breakdown, category_summary, top_po_list


def main():
    print("Loading curated datasets...")
    repl_inputs, fact_inventory = load_inputs()

    print("Building replenishment policy...")
    policy_df = build_replenishment_policy(repl_inputs, fact_inventory)

    print("Building summaries...")
    summary, constraint_breakdown, category_summary, top_po_list = build_replenishment_summaries(policy_df)

    policy_df.to_csv(OUTPUT_DIR / "replenishment_policy_store_sku.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "replenishment_policy_summary.csv", index=False)
    constraint_breakdown.to_csv(OUTPUT_DIR / "replenishment_constraint_breakdown.csv", index=False)
    category_summary.to_csv(OUTPUT_DIR / "replenishment_policy_by_category.csv", index=False)
    top_po_list.to_csv(OUTPUT_DIR / "recommended_po_list_next_cycle.csv", index=False)

    print(f"Done. Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()