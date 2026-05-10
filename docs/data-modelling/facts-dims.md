# Fact and Dimension Tables

Using star schema, we can organize our data into fact and dimension tables.

## Fact Tables Info

Fact tables contain the measures or metrics of a business process. They typically have foreign keys that reference dimension tables and contain numerical data that can be aggregated.

## Dimension Tables Info

Dimension tables contain the attributes or characteristics of the business process. They typically have a primary key that is referenced by the fact table and contain descriptive data that can be used for filtering and grouping in analysis.

## Our Data Model

In our data model, we have the following fact and dimension tables:

### Dimension Tables

#### DimDate

Date_SK (PK)
Full_Date
Year
Quarter
Month_Number, Month_Name
Day_Name, Week_Number

### DimCustomer

Customer_SK (PK)
Customer_ID
Age
Age_Group (derived)
Gender
Country

#### DimSeating

Seating_SK (PK)
Seating_Region
Ticket_Price
Tier_Order (1–4)
4 rows (VIP–Economy)

### Fact Table

Transaction_SK (PK)
Customer_SK (FK)
Date_SK (FK)
Seating_SK (FK)
Num_Tickets
Ticket_Revenue (= Price × Num_Tickets)
Merchandise_Spend
Drink_Spend
Total_Revenue (calculated)
Satisfaction_Score
Recommendation_Likelihood
Repeat_Visit (0/1)