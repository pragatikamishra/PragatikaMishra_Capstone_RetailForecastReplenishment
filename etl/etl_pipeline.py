from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import gdown


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / "source_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Source URLs
# =========================================================
STORES_URL = "https://drive.google.com/uc?id=19oLMbfeF5gJgf_OnHGml7Yp5FsVNGxSZ"
CALENDAR_URL = "https://drive.google.com/uc?id=1g_iQrY975icwAZgSTjT1JRd2B48YVAw7"
PURCHASE_ORDERS_URL = "https://drive.google.com/uc?id=1_c1GeRLtd6R2kZcBszx0zq7A55vq3fGp"
INVENTORY_DAILY_URL = "https://drive.google.com/uc?id=1T01_6WRKo_V0d7GkvymfH5k4f85Pr5Xv"
SALES_DAILY_URL = "https://drive.google.com/uc?id=1A6hftZT4ejuHG5ApGmHey2zks91Z1uMn"
PRODUCTS_URL = "https://drive.google.com/uc?id=1fG-W3PxYt1K-3B-y5lUZHsi5PtbnskrG"


# =========================================================
# Helpers
# =========================================================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def coerce_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def z_from_service_level(service_level: float) -> float:
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


def service_level_by_category(category: str) -> float:
    mapping = {
        "Staples": 0.98,
        "Dairy": 0.97,
        "Fresh": 0.98,
        "Produce": 0.98,
        "Bakery": 0.97,
        "Frozen": 0.96,
        "Beverages": 0.95,
        "Snacks": 0.94,
        "Personal Care": 0.93,
    }
    return mapping.get(str(category).strip().title(), 0.95)


def download_if_needed(url: str, output_path: Path) -> Path:
    if output_path.exists():
        print(f"Using cached file: {output_path.name}")
        return output_path

    print(f"Downloading: {output_path.name}")
    gdown.download(url, str(output_path), quiet=False, fuzzy=True)
    return output_path


def load_csv_from_gdrive(url: str, filename: str) -> pd.DataFrame:
    path = download_if_needed(url, CACHE_DIR / filename)
    return pd.read_csv(path)


def load_json_from_gdrive(url: str, filename: str) -> pd.DataFrame:
    path = download_if_needed(url, CACHE_DIR / filename)
    return pd.read_json(path)


# =========================================================
# Load source data
# =========================================================
def load_data():
    print("Loading source datasets from Google Drive URLs...")

    stores = load_csv_from_gdrive(STORES_URL, "stores.csv")
    calendar = load_csv_from_gdrive(CALENDAR_URL, "calendar.csv")
    purchase_orders = load_csv_from_gdrive(PURCHASE_ORDERS_URL, "purchase_orders.csv")
    inventory_daily = load_csv_from_gdrive(INVENTORY_DAILY_URL, "inventory_daily.csv")
    sales_daily = load_csv_from_gdrive(SALES_DAILY_URL, "sales_daily.csv")
    products = load_json_from_gdrive(PRODUCTS_URL, "products.json")

    stores = coerce_date_columns(standardize_columns(stores))
    calendar = coerce_date_columns(standardize_columns(calendar))
    purchase_orders = coerce_date_columns(standardize_columns(purchase_orders))
    inventory_daily = coerce_date_columns(standardize_columns(inventory_daily))
    sales_daily = coerce_date_columns(standardize_columns(sales_daily))
    products = coerce_date_columns(standardize_columns(products))

    print("Loaded successfully.")
    print(f"stores: {stores.shape}")
    print(f"calendar: {calendar.shape}")
    print(f"purchase_orders: {purchase_orders.shape}")
    print(f"inventory_daily: {inventory_daily.shape}")
    print(f"sales_daily: {sales_daily.shape}")
    print(f"products: {products.shape}")

    return stores, calendar, purchase_orders, inventory_daily, sales_daily, products


# =========================================================
# Clean / standardize
# =========================================================
def clean_data(stores, calendar, purchase_orders, inventory_daily, sales_daily, products):
    stores = stores.drop_duplicates()
    calendar = calendar.drop_duplicates()
    purchase_orders = purchase_orders.drop_duplicates()
    inventory_daily = inventory_daily.drop_duplicates()
    sales_daily = sales_daily.drop_duplicates()
    products = products.drop_duplicates()

    numeric_cols = {
        "calendar": ["promo_flag", "holiday_flag", "is_weekend"],
        "purchase_orders": ["order_qty", "lead_time_days"],
        "inventory_daily": ["on_hand_open", "receipts_units", "on_hand_close"],
        "sales_daily": [
            "units_sold", "true_demand_units", "stockout_censored_units",
            "revenue", "margin_proxy", "promo_flag", "holiday_flag"
        ],
        "products": ["price", "cost", "shelf_life_days", "moq_units"],
    }

    for col in numeric_cols["calendar"]:
        calendar[col] = pd.to_numeric(calendar[col], errors="coerce")

    for col in numeric_cols["purchase_orders"]:
        purchase_orders[col] = pd.to_numeric(purchase_orders[col], errors="coerce")

    for col in numeric_cols["inventory_daily"]:
        inventory_daily[col] = pd.to_numeric(inventory_daily[col], errors="coerce")

    for col in numeric_cols["sales_daily"]:
        sales_daily[col] = pd.to_numeric(sales_daily[col], errors="coerce")

    for col in numeric_cols["products"]:
        products[col] = pd.to_numeric(products[col], errors="coerce")

    products["category"] = products["category"].astype(str).str.strip().str.title()
    stores["region"] = stores["region"].astype(str).str.strip()
    stores["city_tier"] = stores["city_tier"].astype(str).str.strip()
    stores["store_size"] = stores["store_size"].astype(str).str.strip()
    sales_daily["day_of_week"] = sales_daily["day_of_week"].astype(str).str.strip()
    calendar["day_of_week"] = calendar["day_of_week"].astype(str).str.strip()

    sales_daily["units_sold"] = sales_daily["units_sold"].fillna(0).clip(lower=0)
    sales_daily["true_demand_units"] = sales_daily["true_demand_units"].fillna(sales_daily["units_sold"]).clip(lower=0)
    sales_daily["stockout_censored_units"] = sales_daily["stockout_censored_units"].fillna(0).clip(lower=0)

    inventory_daily["on_hand_open"] = inventory_daily["on_hand_open"].fillna(0).clip(lower=0)
    inventory_daily["receipts_units"] = inventory_daily["receipts_units"].fillna(0).clip(lower=0)
    inventory_daily["on_hand_close"] = inventory_daily["on_hand_close"].fillna(0)

    purchase_orders["lead_time_days"] = purchase_orders["lead_time_days"].fillna(7).clip(lower=0)
    purchase_orders["order_qty"] = purchase_orders["order_qty"].fillna(0).clip(lower=0)

    products["moq_units"] = products["moq_units"].fillna(1).clip(lower=1)
    products["price"] = products["price"].fillna(0).clip(lower=0)
    products["cost"] = products["cost"].fillna(0).clip(lower=0)

    return stores, calendar, purchase_orders, inventory_daily, sales_daily, products


# =========================================================
# Curated dataset 1
# =========================================================
def build_fact_sales(sales_daily, stores, products):
    df = sales_daily.merge(products, on="sku_id", how="left")
    df = df.merge(stores, on="store_id", how="left")

    keep_cols = [
        "date",
        "store_id",
        "sku_id",
        "region",
        "city_tier",
        "store_size",
        "category",
        "price",
        "cost",
        "shelf_life_days",
        "moq_units",
        "units_sold",
        "true_demand_units",
        "stockout_censored_units",
        "revenue",
        "margin_proxy",
        "promo_flag",
        "holiday_flag",
        "day_of_week",
    ]

    return df[keep_cols].sort_values(["date", "store_id", "sku_id"]).reset_index(drop=True)


# =========================================================
# Curated dataset 2
# =========================================================
def build_fact_inventory(inventory_daily, sales_daily, stores, products):
    latest_sales_date = sales_daily["date"].max()
    cutoff_date = latest_sales_date - pd.Timedelta(days=27)

    avg_28 = (
        sales_daily[sales_daily["date"] >= cutoff_date]
        .groupby(["store_id", "sku_id"], as_index=False)
        .agg(avg_daily_demand_4w=("true_demand_units", "mean"))
    )

    df = inventory_daily.merge(avg_28, on=["store_id", "sku_id"], how="left")
    df = df.merge(products, on="sku_id", how="left")
    df = df.merge(stores, on="store_id", how="left")

    df["stockout_flag"] = (df["on_hand_close"] <= 0).astype(int)
    df["days_of_cover"] = np.where(
        df["avg_daily_demand_4w"].fillna(0) > 0,
        df["on_hand_close"] / df["avg_daily_demand_4w"],
        np.nan
    )

    keep_cols = [
        "date",
        "store_id",
        "sku_id",
        "region",
        "city_tier",
        "store_size",
        "category",
        "price",
        "cost",
        "shelf_life_days",
        "moq_units",
        "on_hand_open",
        "receipts_units",
        "on_hand_close",
        "stockout_flag",
        "days_of_cover",
    ]

    return df[keep_cols].sort_values(["date", "store_id", "sku_id"]).reset_index(drop=True)


# =========================================================
# Curated dataset 3
# =========================================================
def build_replenishment_inputs(sales_daily, inventory_daily, purchase_orders, stores, products):
    latest_sales_date = sales_daily["date"].max()
    cutoff_date = latest_sales_date - pd.Timedelta(days=55)

    last_56 = sales_daily[sales_daily["date"] >= cutoff_date].copy()

    demand = (
        last_56.groupby(["store_id", "sku_id"], as_index=False)
        .agg(
            avg_daily_demand=("true_demand_units", "mean"),
            demand_std_dev=("true_demand_units", "std"),
        )
    )

    latest_inventory = (
        inventory_daily.sort_values("date")
        .groupby(["store_id", "sku_id"], as_index=False)
        .tail(1)[["store_id", "sku_id", "on_hand_close"]]
        .rename(columns={"on_hand_close": "on_hand_units"})
    )

    lead = (
        purchase_orders.groupby(["store_id", "sku_id"], as_index=False)
        .agg(
            lead_time_days=("lead_time_days", "mean"),
            avg_order_qty=("order_qty", "mean"),
        )
    )

    df = demand.merge(latest_inventory, on=["store_id", "sku_id"], how="left")
    df = df.merge(lead, on=["store_id", "sku_id"], how="left")
    df = df.merge(products, on="sku_id", how="left")
    df = df.merge(stores, on="store_id", how="left")

    df["avg_daily_demand"] = df["avg_daily_demand"].fillna(0)
    df["demand_std_dev"] = df["demand_std_dev"].fillna(0)
    df["lead_time_days"] = df["lead_time_days"].fillna(7)
    df["on_hand_units"] = df["on_hand_units"].fillna(0)
    df["moq_units"] = df["moq_units"].fillna(1).clip(lower=1)

    df["service_level_target"] = df["category"].apply(service_level_by_category)
    df["z_value"] = df["service_level_target"].apply(z_from_service_level)

    df["safety_stock"] = (
        df["z_value"] * df["demand_std_dev"] * np.sqrt(df["lead_time_days"])
    )

    df["reorder_point"] = (
        df["avg_daily_demand"] * df["lead_time_days"] + df["safety_stock"]
    )

    review_cycle_days = 7
    df["order_up_to_level"] = (
        df["avg_daily_demand"] * (df["lead_time_days"] + review_cycle_days) + df["safety_stock"]
    )

    df["recommended_order_qty"] = np.maximum(df["order_up_to_level"] - df["on_hand_units"], 0)

    df["recommended_order_qty"] = np.where(
        (df["recommended_order_qty"] > 0) & (df["recommended_order_qty"] < df["moq_units"]),
        df["moq_units"],
        df["recommended_order_qty"]
    )

    df["max_units_by_shelf_life"] = np.where(
        df["shelf_life_days"].notna() & (df["avg_daily_demand"] > 0),
        df["avg_daily_demand"] * (df["shelf_life_days"] * 0.60),
        np.inf
    )
    df["recommended_order_qty"] = np.minimum(df["recommended_order_qty"], df["max_units_by_shelf_life"])

    keep_cols = [
        "store_id",
        "sku_id",
        "region",
        "city_tier",
        "store_size",
        "category",
        "price",
        "cost",
        "shelf_life_days",
        "moq_units",
        "avg_daily_demand",
        "demand_std_dev",
        "lead_time_days",
        "service_level_target",
        "safety_stock",
        "reorder_point",
        "recommended_order_qty",
        "on_hand_units",
    ]

    return df[keep_cols].sort_values(["store_id", "sku_id"]).reset_index(drop=True)


# =========================================================
# Main
# =========================================================
def main():
    print("Starting ETL pipeline...")
    stores, calendar, purchase_orders, inventory_daily, sales_daily, products = load_data()

    stores, calendar, purchase_orders, inventory_daily, sales_daily, products = clean_data(
        stores, calendar, purchase_orders, inventory_daily, sales_daily, products
    )

    fact_sales = build_fact_sales(sales_daily, stores, products)
    fact_inventory = build_fact_inventory(inventory_daily, sales_daily, stores, products)
    replenishment_inputs = build_replenishment_inputs(
        sales_daily, inventory_daily, purchase_orders, stores, products
    )

    # fact_sales.to_csv(DATA_DIR / "fact_sales_store_sku_daily.csv", index=False)
    # fact_inventory.to_csv(DATA_DIR / "fact_inventory_store_sku_daily.csv", index=False)
    # replenishment_inputs.to_csv(DATA_DIR / "replenishment_inputs_store_sku.csv", index=False)

    # print("ETL completed successfully.")
    # print(DATA_DIR / "fact_sales_store_sku_daily.csv")
    # print(DATA_DIR / "fact_inventory_store_sku_daily.csv")
    # print(DATA_DIR / "replenishment_inputs_store_sku.csv")

    print("Saving fact_sales_store_sku_daily.csv ...")
    fact_sales.to_csv(
    DATA_DIR / "fact_sales_store_sku_daily.csv",
    index=False,
    float_format="%.4f"
    )

    print("Saving fact_inventory_store_sku_daily.csv ...")
    fact_inventory.to_csv(
    DATA_DIR / "fact_inventory_store_sku_daily.csv",
    index=False,
    float_format="%.4f"
    )    

    print("Saving replenishment_inputs_store_sku.csv ...")
    replenishment_inputs.to_csv(
    DATA_DIR / "replenishment_inputs_store_sku.csv",
    index=False,
    float_format="%.4f"
    )

    print("ETL completed successfully.")
    print(DATA_DIR / "fact_sales_store_sku_daily.csv")
    print(DATA_DIR / "fact_inventory_store_sku_daily.csv")
    print(DATA_DIR / "replenishment_inputs_store_sku.csv")



if __name__ == "__main__":
    main()