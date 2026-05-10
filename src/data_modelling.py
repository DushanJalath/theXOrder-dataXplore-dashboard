"""
Transforms the flat table into a star schema:

    DimCustomer
    DimDate
    DimSeating  
    --> FactTransaction

Also returns a fully-joined `analysis` DataFrame for charting convenience.
"""

import pandas as pd


def load_and_prepare(filepath: str = "data/Company_X_Audience.xlsx"):
    """
    Load the raw Excel file and build all star-schema tables.

    Returns: fact, dim_customer, dim_date, dim_seating, analysis (denormalised full join)
    """

    df = pd.read_excel(filepath)
    if df is None or df.empty:
        raise ValueError(f"Failed to load data from {filepath} - check the file path and contents.")
    df["Visit_Date"] = pd.to_datetime(df["Visit_Date"])

    # Customer Dim Table
    dim_customer = (
        df[["Customer_ID", "Age", "Gender", "Country"]]
        .drop_duplicates(subset="Customer_ID")
        .sort_values("Customer_ID")
        .reset_index(drop=True)
    )
    dim_customer.insert(0, "Customer_SK", range(1, len(dim_customer) + 1))
    dim_customer["Age_Group"] = pd.cut(
        dim_customer["Age"],
        bins=[0, 25, 35, 50, 65, 120],
        labels=["Under 25", "25-35", "35-50", "50-65", "65+"],
        right=False,
    ).astype(str)

    # Date Dim Table
    dim_date = (
        df[["Visit_Date"]]
        .drop_duplicates()
        .sort_values("Visit_Date")
        .reset_index(drop=True)
    )
    dim_date.insert(0, "Date_SK", range(1, len(dim_date) + 1))
    dim_date.rename(columns={"Visit_Date": "Full_Date"}, inplace=True)
    dim_date["Year"] = dim_date["Full_Date"].dt.year
    dim_date["Quarter"] = "Q" + dim_date["Full_Date"].dt.quarter.astype(str)
    dim_date["Month_Number"] = dim_date["Full_Date"].dt.month
    dim_date["Month_Name"] = dim_date["Full_Date"].dt.strftime("%B")
    dim_date["Week_Number"] = dim_date["Full_Date"].dt.isocalendar().week.astype(int)
    dim_date["Day_Name"] = dim_date["Full_Date"].dt.strftime("%A")

    # Dim Seating Table
    tier_map = {"VIP": 1, "Premium": 2, "High Economy": 3, "Economy": 4}
    dim_seating = (
        df[["Seating_Region", "Ticket_Price"]]
        .drop_duplicates()
        .assign(Tier_Order=lambda x: x["Seating_Region"].map(tier_map))
        .sort_values("Tier_Order")
        .reset_index(drop=True)
    )
    dim_seating.insert(0, "Seating_SK", range(1, len(dim_seating) + 1))

    # Fact table - grain is one row per transaction (customer visit)
    fact = df.copy()
    fact = fact.merge(dim_customer[["Customer_ID", "Customer_SK"]], on="Customer_ID")
    fact = fact.merge(
        dim_date[["Full_Date", "Date_SK"]],
        left_on="Visit_Date",
        right_on="Full_Date",
    )
    fact = fact.merge(dim_seating[["Seating_Region", "Seating_SK"]], on="Seating_Region")

    # Derived measures (computed once, stored in the fact table)
    fact["Ticket_Revenue"] = fact["Ticket_Price"] * fact["Num_Tickets"]
    fact["Total_Revenue"]  = (
        fact["Ticket_Revenue"] + fact["Merchandise_Spend"] + fact["Drink_Spend"]
    )

    fact = fact.reset_index(drop=True)
    fact.insert(0, "Transaction_SK", range(1, len(fact) + 1))

    # Keep only the fact columns (foreign keys + measures)
    fact = fact[[
        "Transaction_SK",
        "Customer_SK",
        "Date_SK",
        "Seating_SK",
        "Num_Tickets",
        "Ticket_Revenue",
        "Merchandise_Spend",
        "Drink_Spend",
        "Total_Revenue",
        "Satisfaction_Score",
        "Recommendation_Likelihood",
        "Repeat_Visit",
    ]]

    # ── Analysis view (denormalised) ─────────────────────────────────────────
    # Re-join all dimensions for charting. This is the only join used in the
    # dashboard — the star schema tables are the authoritative source of truth.
    analysis = (
        fact
        .merge(dim_customer, on="Customer_SK")
        .merge(dim_date, on="Date_SK")
        .merge(dim_seating, on="Seating_SK")
    )

    return fact, dim_customer, dim_date, dim_seating, analysis