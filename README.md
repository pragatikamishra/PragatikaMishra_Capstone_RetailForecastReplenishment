PragatikaMishra_Capstone_RetailForecastReplenishment/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── fact_sales_store_sku_daily.csv
│   ├── fact_inventory_store_sku_daily.csv
│   └── replenishment_inputs_store_sku.csv
│
├── etl/
│   └── etl_pipeline.py
│
├── analysis/
│   ├── analysis.ipynb
│   └── analysis.py
│
├── dashboard/
│   ├── retail_dashboard.xlsx
│   └── screenshots/
│
└── final_story/
    └── final_memo.pdf\
# How to run the code 
Run main.py to get the results in data folder ans analysis-> output folder

# project overview + north star

# Project Overview
This project builds an end-to-end retail analytics solution to forecast 28-day SKU-store demand, identify stockout and overstock risks, and recommend reorder quantities using a reorder-point and safety-stock based replenishment policy.

Business objective

    Reduce stockouts while controlling excess inventory and holding costs by improving 28-day SKU-store demand forecasting and using a structured replenishment policy.

North star metric

    Fill Rate / Service Level

    Why this is best:

        directly reflects whether customer demand is met

        ties forecasting + inventory decisions together

        leadership can understand it immediately

        improves revenue while reducing stockout pain

# Supporting KPIs

    Fill rate

    Stockout rate

    Lost sales proxy (units)

    Lost sales proxy (revenue)

    WAPE

    MAPE

    Days of inventory on hand

    Overstock rate

    Inventory turns proxy

    Holding cost proxy

# Scope

    Forecast horizon: next 28 days

    Granularity: store-SKU-daily

    Geography: all stores in provided dataset

    Replenishment policy: reorder point + safety stock + order-up-to logic

    Output: curated datasets, risk lists, dashboard, decision memo

# Stakeholder questions

    Which SKUs and stores are most likely to stock out in the next 30 days?

    Which SKUs are overstocked and tying up working capital?

    Which forecast method is most reliable by category?

    How much lost sales can be avoided with the new replenishment plan?

    Which stores and categories are driving most stockout events?

    What reorder quantities should be placed in the next replenishment cycle?

# Forecasting Methods Used
1. Seasonal naive baseline
2. Linear regression using calendar and lag-based demand features

# ETL architecture

Raw files
   ↓
Load + schema standardization
   ↓
Cleaning
- parse dates
- fix casing
- remove duplicates
- handle missing values
- outlier flags
   ↓
Enrichment
- join products
- join stores
- join calendar
- compute revenue / margin proxy
- compute stockout flags / days of cover
   ↓
Feature engineering
- avg daily demand
- demand std dev
- lead times
- service levels
- safety stock
- reorder point
- recommended order qty
   ↓
Export 3 curated datasets



## Analysis Output Files (analysis/output/)
🔹 1. Demand Understanding
demand_summary_by_sku_store.csv

    Avg demand, variability, and demand patterns at SKU-store level

demand_summary_by_category.csv

    Aggregated demand trends by product category

🔹 2. Forecasting Outputs
forecast_metrics_overall.csv

    Overall forecast accuracy (WAPE / MAPE)

forecast_metrics_store_sku.csv

    Forecast accuracy at SKU-store level

forecast_vs_actual_store_sku.csv

    Day-level comparison of forecast vs actual demand

🔹 3. Feature Engineering (optional but useful)
time_features_enriched_sales.csv

    Sales dataset enriched with lag features and rolling averages

🔹 4. Inventory Risk Analysis
inventory_risk_segmented.csv

    SKU-store classified into:

    Stockout Risk

        Healthy

        Overstock Risk

inventory_risk_summary.csv

    Aggregated risk distribution (counts, percentages)

🔹 5. Weekly / Seasonal Analysis
weekly_demand_pattern.csv

    Demand aggregated by day of week

promo_holiday_impact.csv

    Impact of promotions and holidays on demand

🔹 6. Replenishment Inputs (from ETL, used in analysis)
replenishment_inputs_store_sku.csv

    Demand, lead time, safety stock, reorder point

🔹 7. Replenishment Outputs (Part D)
recommended_po_list_next_cycle.csv

    Final recommended purchase orders

replenishment_policy_store_sku.csv

    Policy details:

        Safety stock

        Reorder point

        Order quantity

🔹 8. Impact Estimation (Part E)
impact_projection_summary_by_scenario.csv

    Business impact across:

         Base case

         Best case

         Worst case

    Includes:

        Lost sales avoided

        Fill rate improvement

        Stockout reduction

impact_detailed_store_sku.csv

    SKU-store level impact breakdown