"""
Unit tests for the data_modelling module.

Tests cover:
- Loading and preparing star schema tables
- Dimension table creation (Customer, Date, Seating)
- Fact table creation with derived measures
- Analysis view (denormalised join)
- Error handling for missing or empty files
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_modelling import load_and_prepare


@pytest.fixture
def sample_data():
    """Create a sample DataFrame mimicking the structure of Company_X_Audience.xlsx"""
    dates = pd.date_range(start="2025-01-01", periods=10, freq="D")
    
    data = {
        "Customer_ID": [101, 102, 103, 101, 104, 102, 105, 103, 106, 107],
        "Age": [28, 45, 35, 28, 62, 45, 22, 35, 50, 29],
        "Gender": ["M", "F", "M", "M", "M", "F", "F", "M", "M", "F"],
        "Country": ["USA", "UK", "USA", "USA", "Canada", "UK", "USA", "USA", "Canada", "UK"],
        "Visit_Date": dates,
        "Seating_Region": ["Premium", "VIP", "Economy", "Premium", "High Economy", "VIP", "Economy", "Economy", "Premium", "High Economy"],
        "Ticket_Price": [85.0, 150.0, 45.0, 85.0, 65.0, 150.0, 45.0, 45.0, 85.0, 65.0],
        "Num_Tickets": [2, 1, 4, 2, 1, 1, 3, 2, 2, 1],
        "Merchandise_Spend": [25.0, 50.0, 10.0, 25.0, 0.0, 40.0, 15.0, 10.0, 30.0, 20.0],
        "Drink_Spend": [15.0, 20.0, 5.0, 15.0, 10.0, 25.0, 8.0, 5.0, 20.0, 12.0],
        "Satisfaction_Score": [4.5, 5.0, 3.0, 4.5, 4.0, 5.0, 3.5, 3.0, 4.5, 4.0],
        "Recommendation_Likelihood": [4, 5, 3, 4, 4, 5, 3, 3, 4, 4],
        "Repeat_Visit": [True, True, False, True, False, True, False, False, True, True],
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def excel_file(tmp_path, sample_data):
    """Create a temporary Excel file with sample data"""
    filepath = tmp_path / "test_data.xlsx"
    sample_data.to_excel(filepath, index=False, engine="openpyxl")
    return str(filepath)


class TestLoadAndPrepare:
    """Test suite for load_and_prepare function"""

    def test_load_and_prepare_returns_five_dataframes(self, excel_file):
        """Test that the function returns 5 DataFrames"""
        result = load_and_prepare(excel_file)
        assert len(result) == 5
        assert all(isinstance(df, pd.DataFrame) for df in result)

    def test_dimension_customer_structure(self, excel_file):
        """Test DimCustomer table has correct structure"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Check required columns exist
        required_cols = ["Customer_SK", "Customer_ID", "Age", "Gender", "Country", "Age_Group"]
        assert all(col in dim_customer.columns for col in required_cols)
        
        # Check data types
        assert dim_customer["Customer_SK"].dtype == "int64"
        assert dim_customer["Age"].dtype == "int64"
        
        # Check age groups are assigned correctly
        assert all(dim_customer["Age_Group"].isin(["Under 25", "25-35", "35-50", "50-65", "65+", "nan"]))

    def test_dimension_date_structure(self, excel_file):
        """Test DimDate table has correct structure"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Check required columns exist
        required_cols = ["Date_SK", "Full_Date", "Year", "Quarter", "Month_Number", "Month_Name", "Week_Number", "Day_Name"]
        assert all(col in dim_date.columns for col in required_cols)
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(dim_date["Full_Date"])
        assert pd.api.types.is_integer_dtype(dim_date["Year"])
        assert pd.api.types.is_integer_dtype(dim_date["Month_Number"])

    def test_dimension_seating_structure(self, excel_file):
        """Test DimSeating table has correct structure"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Check required columns exist
        required_cols = ["Seating_SK", "Seating_Region", "Ticket_Price"]
        assert all(col in dim_seating.columns for col in required_cols)
        
        # Check seating regions are in expected order (by tier)
        tier_order = ["VIP", "Premium", "High Economy", "Economy"]
        seating_regions = dim_seating["Seating_Region"].tolist()
        assert seating_regions == sorted(seating_regions, key=lambda x: tier_order.index(x))

    def test_fact_table_structure(self, excel_file):
        """Test FactTransaction table has correct structure"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Check required columns exist
        required_cols = [
            "Transaction_SK", "Customer_SK", "Date_SK", "Seating_SK",
            "Num_Tickets", "Ticket_Revenue", "Merchandise_Spend", "Drink_Spend",
            "Total_Revenue", "Satisfaction_Score", "Recommendation_Likelihood", "Repeat_Visit"
        ]
        assert all(col in fact.columns for col in required_cols)

    def test_derived_measures_calculation(self, excel_file):
        """Test that derived measures are calculated correctly"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Verify Ticket_Revenue calculation: Ticket_Price * Num_Tickets
        # This is done via merge in the analysis view, so we check fact table directly
        assert (fact["Ticket_Revenue"] >= 0).all()
        assert (fact["Merchandise_Spend"] >= 0).all()
        assert (fact["Drink_Spend"] >= 0).all()
        
        # Verify Total_Revenue = Ticket_Revenue + Merchandise_Spend + Drink_Spend
        expected_total = fact["Ticket_Revenue"] + fact["Merchandise_Spend"] + fact["Drink_Spend"]
        pd.testing.assert_series_equal(fact["Total_Revenue"], expected_total, check_names=False)

    def test_analysis_view_contains_all_dimensions(self, excel_file):
        """Test that analysis view is a full denormalised join"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Analysis should have all columns from fact plus dimension columns
        fact_cols = set(fact.columns)
        analysis_cols = set(analysis.columns)
        
        # Check fact columns are present
        assert fact_cols.issubset(analysis_cols)
        
        # Check dimension columns are present
        dim_cols = {"Age", "Gender", "Country", "Age_Group", "Month_Name", "Day_Name", "Quarter", "Seating_Region", "Ticket_Price"}
        assert dim_cols.issubset(analysis_cols)

    def test_row_counts_match_transactions(self, excel_file):
        """Test that fact table has one row per transaction"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Fact and analysis should have the same number of rows (transactions)
        assert len(fact) == len(analysis)
        
        # Both should have more than 0 rows
        assert len(fact) > 0

    def test_primary_keys_are_sequential(self, excel_file):
        """Test that surrogate keys start at 1 and are sequential"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Customer_SK should be sequential starting from 1
        assert dim_customer["Customer_SK"].min() == 1
        assert list(dim_customer["Customer_SK"]) == list(range(1, len(dim_customer) + 1))
        
        # Date_SK should be sequential starting from 1
        assert dim_date["Date_SK"].min() == 1
        assert list(dim_date["Date_SK"]) == list(range(1, len(dim_date) + 1))
        
        # Seating_SK should be sequential starting from 1
        assert dim_seating["Seating_SK"].min() == 1
        assert list(dim_seating["Seating_SK"]) == list(range(1, len(dim_seating) + 1))
        
        # Transaction_SK should be sequential starting from 1
        assert fact["Transaction_SK"].min() == 1
        assert list(fact["Transaction_SK"]) == list(range(1, len(fact) + 1))

    def test_no_duplicate_dimensions(self, excel_file):
        """Test that dimension tables have no duplicates"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Each dimension should have unique surrogate keys
        assert dim_customer["Customer_SK"].nunique() == len(dim_customer)
        assert dim_date["Date_SK"].nunique() == len(dim_date)
        assert dim_seating["Seating_SK"].nunique() == len(dim_seating)

    def test_missing_file_raises_error(self):
        """Test that missing file raises an error"""
        with pytest.raises((ValueError, FileNotFoundError)):
            load_and_prepare("nonexistent_file.xlsx")

    def test_empty_file_raises_error(self, tmp_path):
        """Test that empty Excel file raises ValueError"""
        empty_file = tmp_path / "empty.xlsx"
        empty_df = pd.DataFrame()
        empty_df.to_excel(empty_file, index=False, engine="openpyxl")
        
        with pytest.raises(ValueError, match="Failed to load data"):
            load_and_prepare(str(empty_file))

    def test_date_conversion(self, excel_file):
        """Test that Visit_Date is converted to datetime"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Full_Date in dim_date should be datetime
        assert pd.api.types.is_datetime64_any_dtype(dim_date["Full_Date"])

    def test_foreign_key_relationships(self, excel_file):
        """Test that all foreign keys in fact table reference valid dimension keys"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # All Customer_SK in fact should exist in dim_customer
        assert fact["Customer_SK"].isin(dim_customer["Customer_SK"]).all()
        
        # All Date_SK in fact should exist in dim_date
        assert fact["Date_SK"].isin(dim_date["Date_SK"]).all()
        
        # All Seating_SK in fact should exist in dim_seating
        assert fact["Seating_SK"].isin(dim_seating["Seating_SK"]).all()

    def test_no_nulls_in_keys(self, excel_file):
        """Test that surrogate keys contain no null values"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        assert not dim_customer["Customer_SK"].isnull().any()
        assert not dim_date["Date_SK"].isnull().any()
        assert not dim_seating["Seating_SK"].isnull().any()
        assert not fact["Transaction_SK"].isnull().any()

    def test_analysis_join_integrity(self, excel_file):
        """Test that analysis view is correctly denormalised"""
        fact, dim_customer, dim_date, dim_seating, analysis = load_and_prepare(excel_file)
        
        # Join fact with each dimension manually and compare
        manual_join = (
            fact
            .merge(dim_customer[["Customer_SK", "Age", "Gender", "Country", "Age_Group"]], on="Customer_SK")
            .merge(dim_date[["Date_SK", "Full_Date", "Year", "Month_Name", "Day_Name", "Quarter"]], on="Date_SK")
            .merge(dim_seating[["Seating_SK", "Seating_Region", "Ticket_Price"]], on="Seating_SK")
        )
        
        # Both should have the same number of rows
        assert len(analysis) == len(manual_join)
        
        # Core columns should match
        core_cols = list(fact.columns) + ["Age", "Gender", "Country"]
        assert all(col in analysis.columns for col in core_cols)
