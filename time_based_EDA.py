"""
============================================================
  CM 3430 / BI Dashboard Project
  Time-Based EDA — Company X Audience Dataset
  Author : Your Team
  Dataset: 800 concert-venue customer records (Jan–Sep 2025)
============================================================

Sections
--------
0.  Setup & Data Loading
1.  Date Engineering (feature extraction)
2.  Monthly Visit Volume & Traffic Trends
3.  Monthly Revenue Trends (Ticket / Merch / Drink / Total)
4.  Revenue Mix Over Time (stacked area)
5.  Seating-Region Performance Over Time
6.  Repeat-Visit Behaviour Across Months
7.  Satisfaction & Recommendation Trends Over Time
8.  Weekly & Day-of-Week Patterns
9.  Cumulative Revenue Growth
10. Rolling Averages (smoothed KPIs)
11. Month-over-Month Growth Rates
12. Heatmap Calendar View (visits per day)
13. Summary Statistics Table by Month
"""

# ─────────────────────────────────────────────
# 0. Setup & Data Loading
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates  as mdates
import seaborn           as sns
import warnings
warnings.filterwarnings("ignore")

# ── Seaborn / Matplotlib global theme ──────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi"       : 130,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "xtick.labelsize"  : 9,
    "ytick.labelsize"  : 9,
    "legend.fontsize"  : 9,
    "figure.facecolor" : "white",
})

PALETTE_REGION = {
    "VIP"          : "#2c7bb6",
    "Premium"      : "#abd9e9",
    "High Economy" : "#fdae61",
    "Economy"      : "#d7191c",
}

# ── Load dataset ───────────────────────────────────────────────────────────
df = pd.read_excel("Company_X_Audience.xlsx")          # adjust path if needed
# If running from a different directory:
# df = pd.read_excel("/path/to/Company_X_Audience.xlsx")

print("=" * 60)
print("Dataset shape :", df.shape)
print("Date range    :", df["Visit_Date"].min().date(),
      "→", df["Visit_Date"].max().date())
print("=" * 60)


# ─────────────────────────────────────────────
# 1. Date Engineering
# ─────────────────────────────────────────────
df["Visit_Date"]  = pd.to_datetime(df["Visit_Date"])

df["Year"]        = df["Visit_Date"].dt.year
df["Month"]       = df["Visit_Date"].dt.month                    # 1–12
df["Month_Name"]  = df["Visit_Date"].dt.strftime("%b")           # Jan, Feb …
df["Month_Period"]= df["Visit_Date"].dt.to_period("M")           # 2025-01 …
df["Week"]        = df["Visit_Date"].dt.isocalendar().week.astype(int)
df["DayOfWeek"]   = df["Visit_Date"].dt.dayofweek               # 0=Mon … 6=Sun
df["DayName"]     = df["Visit_Date"].dt.day_name()
df["Quarter"]     = df["Visit_Date"].dt.quarter.map(
                        {1:"Q1", 2:"Q2", 3:"Q3", 4:"Q4"})

# Derived revenue columns
df["Ticket_Revenue"]= df["Ticket_Price"] * df["Num_Tickets"]
df["Total_Revenue"] = (df["Ticket_Revenue"]
                       + df["Merchandise_Spend"]
                       + df["Drink_Spend"])

# Ordered month label list (for consistent x-axis)
MONTH_ORDER = (df.groupby("Month_Period")["Month_Name"]
               .first()
               .sort_index()
               .tolist())

print("\nEngineered columns preview:")
print(df[["Visit_Date","Month_Name","DayName","Quarter",
          "Ticket_Revenue","Total_Revenue"]].head(5).to_string(index=False))


# ─────────────────────────────────────────────
# 2. Monthly Visit Volume & Traffic Trends
# ─────────────────────────────────────────────
monthly_visits = (df.groupby("Month_Period")
                    .agg(Visits=("Customer_ID","count"))
                    .reset_index())
monthly_visits["Month_Name"] = monthly_visits["Month_Period"].dt.strftime("%b")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
fig.suptitle("Section 2 — Monthly Visit Volume & Traffic Trends", fontsize=14, fontweight="bold")

# 2a – Bar chart
ax = axes[0]
bars = ax.bar(monthly_visits["Month_Name"], monthly_visits["Visits"],
              color=sns.color_palette("Blues_d", len(monthly_visits)), edgecolor="white")
ax.bar_label(bars, fmt="%d", label_type="edge", fontsize=8)
ax.set_title("Total Visits per Month")
ax.set_xlabel("Month (2025)")
ax.set_ylabel("Number of Visits")
ax.set_ylim(0, monthly_visits["Visits"].max() * 1.15)

# 2b – Line chart with trend
ax2 = axes[1]
ax2.plot(monthly_visits["Month_Name"], monthly_visits["Visits"],
         marker="o", linewidth=2, color="#2c7bb6", label="Monthly Visits")
# Add trend line
z = np.polyfit(range(len(monthly_visits)), monthly_visits["Visits"], 1)
p = np.poly1d(z)
ax2.plot(monthly_visits["Month_Name"],
         p(range(len(monthly_visits))),
         "--", color="#d7191c", linewidth=1.5, label="Trend")
ax2.fill_between(monthly_visits["Month_Name"],
                 monthly_visits["Visits"], alpha=0.15, color="#2c7bb6")
ax2.set_title("Visit Trend Line")
ax2.set_xlabel("Month (2025)")
ax2.set_ylabel("Number of Visits")
ax2.legend()

plt.tight_layout()
plt.savefig("plot_02_monthly_visits.png", bbox_inches="tight")
plt.show()
print("\n[2] Monthly visits saved → plot_02_monthly_visits.png")


# ─────────────────────────────────────────────
# 3. Monthly Revenue Trends
# ─────────────────────────────────────────────
monthly_rev = (df.groupby("Month_Period")
                 .agg(Ticket_Rev  =("Ticket_Revenue",      "sum"),
                      Merch_Rev   =("Merchandise_Spend",   "sum"),
                      Drink_Rev   =("Drink_Spend",         "sum"),
                      Total_Rev   =("Total_Revenue",       "sum"),
                      Avg_Rev_Per_Visit=("Total_Revenue",  "mean"))
                 .reset_index())
monthly_rev["Month_Name"] = monthly_rev["Month_Period"].dt.strftime("%b")

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle("Section 3 — Monthly Revenue Trends", fontsize=14, fontweight="bold")

streams = [
    ("Ticket_Rev",  "Ticket Revenue",         "#2c7bb6"),
    ("Merch_Rev",   "Merchandise Revenue",    "#1a9641"),
    ("Drink_Rev",   "Drink Revenue",          "#fdae61"),
    ("Total_Rev",   "Total Revenue",          "#d7191c"),
]
for ax, (col, title, color) in zip(axes.flat, streams):
    ax.plot(monthly_rev["Month_Name"], monthly_rev[col],
            marker="o", linewidth=2.2, color=color)
    ax.fill_between(monthly_rev["Month_Name"], monthly_rev[col],
                    alpha=0.12, color=color)
    ax.set_title(title)
    ax.set_xlabel("Month (2025)")
    ax.set_ylabel("USD ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("plot_03_monthly_revenue.png", bbox_inches="tight")
plt.show()
print("[3] Revenue trends saved → plot_03_monthly_revenue.png")


# ─────────────────────────────────────────────
# 4. Revenue Mix Over Time (Stacked Area)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Section 4 — Revenue Mix Over Time", fontsize=14, fontweight="bold")

# 4a – Stacked area (absolute)
ax = axes[0]
ax.stackplot(monthly_rev["Month_Name"],
             monthly_rev["Ticket_Rev"],
             monthly_rev["Merch_Rev"],
             monthly_rev["Drink_Rev"],
             labels=["Ticket", "Merchandise", "Drink"],
             colors=["#2c7bb6", "#1a9641", "#fdae61"],
             alpha=0.85)
ax.set_title("Stacked Revenue by Stream (Absolute)")
ax.set_xlabel("Month (2025)")
ax.set_ylabel("USD ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend(loc="upper left")

# 4b – 100% stacked area (proportion)
ax2 = axes[1]
total_ = monthly_rev[["Ticket_Rev","Merch_Rev","Drink_Rev"]].sum(axis=1)
tick_pct  = monthly_rev["Ticket_Rev"]  / total_ * 100
merch_pct = monthly_rev["Merch_Rev"]   / total_ * 100
drink_pct = monthly_rev["Drink_Rev"]   / total_ * 100
ax2.stackplot(monthly_rev["Month_Name"],
              tick_pct, merch_pct, drink_pct,
              labels=["Ticket %", "Merchandise %", "Drink %"],
              colors=["#2c7bb6","#1a9641","#fdae61"],
              alpha=0.85)
ax2.set_title("Revenue Share by Stream (%) ")
ax2.set_xlabel("Month (2025)")
ax2.set_ylabel("Share (%)")
ax2.set_ylim(0, 100)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("plot_04_revenue_mix.png", bbox_inches="tight")
plt.show()
print("[4] Revenue mix saved → plot_04_revenue_mix.png")


# ─────────────────────────────────────────────
# 5. Seating-Region Performance Over Time
# ─────────────────────────────────────────────
region_month = (df.groupby(["Month_Period","Seating_Region"])
                  .agg(Total_Rev  =("Total_Revenue","sum"),
                       Visits     =("Customer_ID",  "count"),
                       Avg_Ticket =("Ticket_Price", "mean"))
                  .reset_index())
region_month["Month_Name"] = region_month["Month_Period"].dt.strftime("%b")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Section 5 — Seating-Region Performance Over Time",
             fontsize=14, fontweight="bold")

metrics = [
    ("Total_Rev",   "Total Revenue (USD)"),
    ("Visits",      "Visits (count)"),
    ("Avg_Ticket",  "Avg Ticket Price (USD)"),
]
for ax, (col, ylabel) in zip(axes, metrics):
    for region, grp in region_month.groupby("Seating_Region"):
        ax.plot(grp["Month_Name"], grp[col],
                marker="o", label=region,
                color=PALETTE_REGION[region], linewidth=2)
    ax.set_title(ylabel)
    ax.set_xlabel("Month (2025)")
    ax.set_ylabel(ylabel)
    if col in ("Total_Rev","Avg_Ticket"):
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}"))
    ax.legend(title="Region", fontsize=8)

plt.tight_layout()
plt.savefig("plot_05_seating_region_time.png", bbox_inches="tight")
plt.show()
print("[5] Seating-region trends saved → plot_05_seating_region_time.png")


# ─────────────────────────────────────────────
# 6. Repeat-Visit Behaviour Across Months
# ─────────────────────────────────────────────
repeat_month = (df.groupby(["Month_Period","Repeat_Visit"])
                  .agg(Visits=("Customer_ID","count"),
                       Avg_Total_Rev=("Total_Revenue","mean"),
                       Avg_Sat=("Satisfaction_Score","mean"))
                  .reset_index())
repeat_month["Month_Name"] = repeat_month["Month_Period"].dt.strftime("%b")
repeat_month["Label"] = repeat_month["Repeat_Visit"].map(
    {0:"First-Time", 1:"Repeat"})

# Pivot for easy plotting
piv_visits = repeat_month.pivot(
    index="Month_Name", columns="Label", values="Visits").fillna(0)
piv_rev    = repeat_month.pivot(
    index="Month_Name", columns="Label", values="Avg_Total_Rev").fillna(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Section 6 — Repeat vs First-Time Visit Behaviour Over Time",
             fontsize=14, fontweight="bold")

# 6a – Visit counts stacked bar
piv_visits.loc[MONTH_ORDER].plot(kind="bar", stacked=True,
    color=["#fdae61","#2c7bb6"], ax=axes[0], rot=0)
axes[0].set_title("Visit Volume by Visitor Type")
axes[0].set_xlabel("Month (2025)")
axes[0].set_ylabel("Number of Visits")
axes[0].legend(title="Type")

# 6b – Repeat rate % line
repeat_rate = (df.groupby("Month_Period")["Repeat_Visit"]
                 .mean() * 100).reset_index()
repeat_rate["Month_Name"] = repeat_rate["Month_Period"].dt.strftime("%b")
axes[1].plot(repeat_rate["Month_Name"], repeat_rate["Repeat_Visit"],
             marker="o", linewidth=2, color="#2c7bb6", label="Repeat Rate %")
axes[1].axhline(repeat_rate["Repeat_Visit"].mean(), linestyle="--",
                color="#d7191c", linewidth=1.4,
                label=f'Mean {repeat_rate["Repeat_Visit"].mean():.1f}%')
axes[1].set_title("Monthly Repeat-Visit Rate (%)")
axes[1].set_xlabel("Month (2025)")
axes[1].set_ylabel("Repeat Visit Rate (%)")
axes[1].legend()

# 6c – Avg revenue: first-time vs repeat
for label, color in [("First-Time","#fdae61"),("Repeat","#2c7bb6")]:
    data = piv_rev[label].loc[MONTH_ORDER]
    axes[2].plot(MONTH_ORDER, data, marker="o",
                 linewidth=2, label=label, color=color)
axes[2].set_title("Avg Total Revenue: First-Time vs Repeat")
axes[2].set_xlabel("Month (2025)")
axes[2].set_ylabel("Avg Revenue (USD)")
axes[2].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[2].legend(title="Type")

plt.tight_layout()
plt.savefig("plot_06_repeat_visit_time.png", bbox_inches="tight")
plt.show()
print("[6] Repeat-visit trends saved → plot_06_repeat_visit_time.png")


# ─────────────────────────────────────────────
# 7. Satisfaction & Recommendation Trends
# ─────────────────────────────────────────────
exp_month = (df.groupby("Month_Period")
               .agg(Avg_Sat  =("Satisfaction_Score",       "mean"),
                    Avg_Rec  =("Recommendation_Likelihood","mean"),
                    Std_Sat  =("Satisfaction_Score",       "std"))
               .reset_index())
exp_month["Month_Name"] = exp_month["Month_Period"].dt.strftime("%b")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Section 7 — Customer Experience Trends Over Time",
             fontsize=14, fontweight="bold")

# 7a – Satisfaction with confidence band
ax = axes[0]
ax.plot(exp_month["Month_Name"], exp_month["Avg_Sat"],
        marker="o", linewidth=2, color="#2c7bb6", label="Avg Satisfaction")
ax.fill_between(exp_month["Month_Name"],
                exp_month["Avg_Sat"] - exp_month["Std_Sat"],
                exp_month["Avg_Sat"] + exp_month["Std_Sat"],
                alpha=0.15, color="#2c7bb6", label="±1 Std Dev")
ax.axhline(df["Satisfaction_Score"].mean(), linestyle="--",
           color="#d7191c", linewidth=1.4,
           label=f'Overall Mean ({df["Satisfaction_Score"].mean():.2f})')
ax.set_title("Monthly Avg Satisfaction Score")
ax.set_xlabel("Month (2025)")
ax.set_ylabel("Score (1–10)")
ax.set_ylim(0, 10)
ax.legend()

# 7b – Recommendation likelihood
ax2 = axes[1]
ax2.plot(exp_month["Month_Name"], exp_month["Avg_Rec"],
         marker="s", linewidth=2, color="#1a9641", label="Avg Recommendation")
ax2.axhline(df["Recommendation_Likelihood"].mean(), linestyle="--",
            color="#d7191c", linewidth=1.4,
            label=f'Overall Mean ({df["Recommendation_Likelihood"].mean():.2f})')
ax2.fill_between(exp_month["Month_Name"], exp_month["Avg_Rec"],
                 alpha=0.12, color="#1a9641")
ax2.set_title("Monthly Avg Recommendation Likelihood")
ax2.set_xlabel("Month (2025)")
ax2.set_ylabel("Score (0–10)")
ax2.set_ylim(0, 10)
ax2.legend()

plt.tight_layout()
plt.savefig("plot_07_experience_trends.png", bbox_inches="tight")
plt.show()
print("[7] Experience trends saved → plot_07_experience_trends.png")


# ─────────────────────────────────────────────
# 8. Weekly & Day-of-Week Patterns
# ─────────────────────────────────────────────
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

dow_stats = (df.groupby("DayName")
               .agg(Visits=("Customer_ID","count"),
                    Avg_Rev=("Total_Revenue","mean"),
                    Avg_Sat=("Satisfaction_Score","mean"))
               .reindex(dow_order)
               .reset_index())

weekly_rev = (df.groupby("Week")
                .agg(Total_Rev=("Total_Revenue","sum"),
                     Visits=("Customer_ID","count"))
                .reset_index())

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Section 8 — Weekly & Day-of-Week Patterns",
             fontsize=14, fontweight="bold")

# 8a – Visits by day of week
colors_dow = sns.color_palette("Blues_d", 7)
axes[0].bar(dow_stats["DayName"], dow_stats["Visits"],
            color=colors_dow, edgecolor="white")
axes[0].set_title("Visits by Day of Week")
axes[0].set_xlabel("Day")
axes[0].set_ylabel("Total Visits")
axes[0].tick_params(axis="x", rotation=30)

# 8b – Avg revenue by day of week
axes[1].bar(dow_stats["DayName"], dow_stats["Avg_Rev"],
            color=sns.color_palette("Greens_d", 7), edgecolor="white")
axes[1].set_title("Avg Revenue per Visit by Day")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Avg Revenue (USD)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# 8c – Weekly total revenue trend
axes[2].plot(weekly_rev["Week"], weekly_rev["Total_Rev"],
             marker="o", linewidth=1.8, color="#2c7bb6")
axes[2].fill_between(weekly_rev["Week"], weekly_rev["Total_Rev"],
                     alpha=0.15, color="#2c7bb6")
axes[2].set_title("Total Revenue by Calendar Week")
axes[2].set_xlabel("ISO Week Number")
axes[2].set_ylabel("Total Revenue (USD)")
axes[2].yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("plot_08_weekly_dow.png", bbox_inches="tight")
plt.show()
print("[8] Weekly/DOW patterns saved → plot_08_weekly_dow.png")


# ─────────────────────────────────────────────
# 9. Cumulative Revenue Growth
# ─────────────────────────────────────────────
daily_rev = (df.groupby("Visit_Date")
               .agg(Daily_Rev=("Total_Revenue","sum"),
                    Daily_Visits=("Customer_ID","count"))
               .reset_index()
               .sort_values("Visit_Date"))
daily_rev["Cum_Rev"]    = daily_rev["Daily_Rev"].cumsum()
daily_rev["Cum_Visits"] = daily_rev["Daily_Visits"].cumsum()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Section 9 — Cumulative Growth Over Time",
             fontsize=14, fontweight="bold")

ax = axes[0]
ax.plot(daily_rev["Visit_Date"], daily_rev["Cum_Rev"],
        linewidth=2, color="#d7191c")
ax.fill_between(daily_rev["Visit_Date"], daily_rev["Cum_Rev"],
                alpha=0.12, color="#d7191c")
ax.set_title("Cumulative Total Revenue (YTD)")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Revenue (USD)")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

ax2 = axes[1]
ax2.plot(daily_rev["Visit_Date"], daily_rev["Cum_Visits"],
         linewidth=2, color="#2c7bb6")
ax2.fill_between(daily_rev["Visit_Date"], daily_rev["Cum_Visits"],
                 alpha=0.12, color="#2c7bb6")
ax2.set_title("Cumulative Visitor Count (YTD)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Cumulative Visitors")
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

plt.tight_layout()
plt.savefig("plot_09_cumulative.png", bbox_inches="tight")
plt.show()
print("[9] Cumulative growth saved → plot_09_cumulative.png")


# ─────────────────────────────────────────────
# 10. Rolling Averages (7-day & 14-day)
# ─────────────────────────────────────────────
daily_rev["Roll7_Rev"]  = daily_rev["Daily_Rev"].rolling(7,  min_periods=1).mean()
daily_rev["Roll14_Rev"] = daily_rev["Daily_Rev"].rolling(14, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(daily_rev["Visit_Date"], daily_rev["Daily_Rev"],
       color="#d3e9f7", label="Daily Revenue", width=1)
ax.plot(daily_rev["Visit_Date"], daily_rev["Roll7_Rev"],
        linewidth=2, color="#2c7bb6", label="7-day Rolling Avg")
ax.plot(daily_rev["Visit_Date"], daily_rev["Roll14_Rev"],
        linewidth=2, color="#d7191c", linestyle="--", label="14-day Rolling Avg")
ax.set_title("Section 10 — Daily Revenue with Rolling Averages", fontsize=13, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue (USD)")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.legend()
plt.tight_layout()
plt.savefig("plot_10_rolling_avg.png", bbox_inches="tight")
plt.show()
print("[10] Rolling averages saved → plot_10_rolling_avg.png")


# ─────────────────────────────────────────────
# 11. Month-over-Month Growth Rates
# ─────────────────────────────────────────────
monthly_rev["Rev_MoM_%"]    = monthly_rev["Total_Rev"].pct_change()    * 100
monthly_rev["Visit_MoM_%"]  = monthly_rev["Month_Name"].map(
    monthly_visits.set_index("Month_Name")["Visits"]).pct_change() * 100

# Recompute visits properly
monthly_visits_indexed = monthly_visits.set_index("Month_Period")["Visits"]
monthly_rev["Visits"]       = monthly_rev["Month_Period"].map(monthly_visits_indexed)
monthly_rev["Visit_MoM_%"]  = monthly_rev["Visits"].pct_change() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Section 11 — Month-over-Month (MoM) Growth Rates",
             fontsize=14, fontweight="bold")

def mom_bar(ax, series, month_names, title, unit="%"):
    colors = ["#1a9641" if v >= 0 else "#d7191c" for v in series.fillna(0)]
    ax.bar(month_names[1:], series.dropna(), color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Month (2025)")
    ax.set_ylabel(f"MoM Change ({unit})")
    for i, v in enumerate(series.dropna()):
        ax.text(i, v + (0.5 if v >= 0 else -1.5),
                f"{v:+.1f}%", ha="center", fontsize=8,
                color="white" if abs(v) > 3 else "black")

mom_bar(axes[0], monthly_rev["Rev_MoM_%"],
        monthly_rev["Month_Name"], "Total Revenue MoM Growth")
mom_bar(axes[1], monthly_rev["Visit_MoM_%"],
        monthly_rev["Month_Name"], "Visit Volume MoM Growth")

plt.tight_layout()
plt.savefig("plot_11_mom_growth.png", bbox_inches="tight")
plt.show()
print("[11] MoM growth rates saved → plot_11_mom_growth.png")


# ─────────────────────────────────────────────
# 12. Heatmap Calendar View (visits per day)
# ─────────────────────────────────────────────
pivot_cal = (df.groupby(["Month","DayOfWeek"])["Customer_ID"]
               .count()
               .unstack(fill_value=0))
pivot_cal.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
pivot_cal.index   = [pd.Timestamp(f"2025-{m:02d}-01").strftime("%B")
                     for m in pivot_cal.index]

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(pivot_cal, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label":"Visit Count"}, ax=ax)
ax.set_title("Section 12 — Visits Heatmap: Month × Day-of-Week",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig("plot_12_heatmap_calendar.png", bbox_inches="tight")
plt.show()
print("[12] Calendar heatmap saved → plot_12_heatmap_calendar.png")


# ─────────────────────────────────────────────
# 13. Summary Statistics Table by Month
# ─────────────────────────────────────────────
summary = (df.groupby("Month_Name")
             .agg(
                Visits          =("Customer_ID",              "count"),
                Total_Revenue   =("Total_Revenue",            "sum"),
                Avg_Rev_Visit   =("Total_Revenue",            "mean"),
                Ticket_Revenue  =("Ticket_Revenue",           "sum"),
                Merch_Revenue   =("Merchandise_Spend",        "sum"),
                Drink_Revenue   =("Drink_Spend",              "sum"),
                Repeat_Rate_pct =("Repeat_Visit",             "mean"),
                Avg_Satisfaction=("Satisfaction_Score",       "mean"),
                Avg_Recommend   =("Recommendation_Likelihood","mean"),
             )
             .reset_index())

# Reorder by calendar month
summary["Sort"] = pd.Categorical(
    summary["Month_Name"], categories=MONTH_ORDER, ordered=True)
summary = summary.sort_values("Sort").drop(columns="Sort")

# Format
summary["Total_Revenue"]  = summary["Total_Revenue"].map("${:,.0f}".format)
summary["Avg_Rev_Visit"]  = summary["Avg_Rev_Visit"].map("${:,.2f}".format)
summary["Ticket_Revenue"] = summary["Ticket_Revenue"].map("${:,.0f}".format)
summary["Merch_Revenue"]  = summary["Merch_Revenue"].map("${:,.0f}".format)
summary["Drink_Revenue"]  = summary["Drink_Revenue"].map("${:,.0f}".format)
summary["Repeat_Rate_pct"]= (summary["Repeat_Rate_pct"]*100).map("{:.1f}%".format)
summary["Avg_Satisfaction"]= summary["Avg_Satisfaction"].map("{:.2f}".format)
summary["Avg_Recommend"]  = summary["Avg_Recommend"].map("{:.2f}".format)

print("\n" + "=" * 100)
print("Section 13 — Monthly Summary Statistics")
print("=" * 100)
print(summary.to_string(index=False))
print("=" * 100)

# Save summary to CSV
summary.to_csv("monthly_summary_stats.csv", index=False)
print("\n[13] Summary table saved → monthly_summary_stats.csv")


# ─────────────────────────────────────────────
# Bonus: Quarterly Roll-up
# ─────────────────────────────────────────────
quarterly = (df.groupby("Quarter")
               .agg(Visits        =("Customer_ID",              "count"),
                    Total_Revenue  =("Total_Revenue",            "sum"),
                    Avg_Sat        =("Satisfaction_Score",       "mean"),
                    Repeat_Rate    =("Repeat_Visit",             "mean"))
               .reset_index())
quarterly["Repeat_Rate"] *= 100

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Bonus — Quarterly Roll-up KPIs", fontsize=14, fontweight="bold")

q_colors = ["#2c7bb6","#1a9641","#fdae61","#d7191c"]

for ax, (col, title) in zip(axes, [
    ("Total_Revenue","Total Revenue (USD)"),
    ("Visits",       "Total Visits"),
    ("Repeat_Rate",  "Repeat Visit Rate (%)")
]):
    bars = ax.bar(quarterly["Quarter"], quarterly[col],
                  color=q_colors[:len(quarterly)], edgecolor="white", width=0.5)
    ax.bar_label(bars,
                 labels=([f"${v:,.0f}" for v in quarterly[col]]
                         if col == "Total_Revenue"
                         else [f"{v:.1f}%" if col == "Repeat_Rate"
                               else f"{int(v)}" for v in quarterly[col]]),
                 label_type="edge", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(title)
    if col == "Total_Revenue":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

plt.tight_layout()
plt.savefig("plot_bonus_quarterly.png", bbox_inches="tight")
plt.show()
print("[Bonus] Quarterly KPIs saved → plot_bonus_quarterly.png")

print("\n" + "=" * 60)
print("  All Time-Based EDA plots generated successfully!")
print("  Output files:")
for i, f in enumerate([
    "plot_02_monthly_visits.png",
    "plot_03_monthly_revenue.png",
    "plot_04_revenue_mix.png",
    "plot_05_seating_region_time.png",
    "plot_06_repeat_visit_time.png",
    "plot_07_experience_trends.png",
    "plot_08_weekly_dow.png",
    "plot_09_cumulative.png",
    "plot_10_rolling_avg.png",
    "plot_11_mom_growth.png",
    "plot_12_heatmap_calendar.png",
    "monthly_summary_stats.csv",
    "plot_bonus_quarterly.png",
], start=2):
    print(f"   → {f}")
print("=" * 60)