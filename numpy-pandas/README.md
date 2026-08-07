# numpy-pandas

This folder contains practice material for learning NumPy and pandas. The scripts are mostly commented walkthroughs, so they act as a reference library of examples rather than standalone runnable programs.

## Folder Contents

### `1.py`
NumPy basics and array operations.

Topics covered in the file:
- importing NumPy
- creating arrays from lists
- `zeros`, `ones`, `arange`, and `linspace`
- array properties such as dimensions, shape, and size
- indexing and slicing
- filtering with conditions
- reshaping arrays
- `ravel()` and `flatten()`
- broadcasting
- aggregate functions such as sum, mean, median, max, min, std, and var
- appending, inserting, and deleting values
- searching with `where()`
- `unique()`
- concatenation
- vertical and horizontal stacking

This file is mainly a NumPy cheat sheet with examples for common array operations.

### `2.py`
Pandas DataFrame creation and basic data manipulation.

Topics covered in the file:
- importing pandas
- creating a DataFrame from a dictionary
- reading CSV files
- writing data to CSV and Excel
- previewing data with `head()` and `tail()`
- checking shape, columns, index, and dtypes
- `info()` and `describe()`
- conditional selection
- updating values with `loc[]`
- adding a computed column such as `bonus`
- inserting a new column with `insert()`
- updating a single cell by index
- dropping columns with `drop()`

This script also shows how the sample employee data can be exported to:
- `Employee data.csv`
- `employee data.xlsx`

### `3.py`
Pandas data cleaning, sorting, grouping, and merging.

Topics covered in the file:
- reading a CSV dataset
- checking for missing values with `isnull()`
- removing rows with `dropna()`
- filling missing values with `fillna()`
- interpolation for missing numeric data
- sorting values with `sort_values()`
- grouping and aggregating with `groupby()`
- merging datasets with `merge()`
- inner join, right join, and cross join examples

This file focuses on common preprocessing tasks used before analysis or machine learning.

### `data.csv`
An employee-style dataset used for pandas practice.

Columns:
- `Employee_ID`
- `Name`
- `Age`
- `Department`
- `City`
- `Salary`
- `Experience`
- `Performance`

This file includes missing values in several columns so it can be used for data-cleaning exercises.

### `Employee data.csv`
A small sample employee dataset with the columns:
- `Name`
- `Age`
- `city`

This is the simple CSV used in the pandas example in `2.py`.

### `employee data.xlsx`
The Excel version of the sample employee dataset generated from `2.py`.

### `employees.csv`
A larger employee dataset for pandas practice.

Columns:
- `First Name`
- `Gender`
- `Start Date`
- `Last Login Time`
- `Salary`
- `Bonus %`
- `Senior Management`
- `Team`

This dataset contains missing values and mixed data types, which makes it useful for cleaning, filtering, grouping, and merging examples.

## Suggested Usage

If you want to follow the folder in order:
1. Start with `1.py` to learn NumPy array basics.
2. Move to `2.py` for pandas DataFrame creation and editing.
3. Use `3.py` for cleaning, sorting, grouping, and merging datasets.
4. Practice on `data.csv` and `employees.csv` after you understand the examples.
