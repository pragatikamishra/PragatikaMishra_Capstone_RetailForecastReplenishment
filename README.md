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


# How to Run ETL
```bash
python etl/etl_pipeline.py