import pandas as pd

# df=pd.read_csv('data.csv' )

# print(df)

# finding mising data
# isnull() method to check for missing values
# print(df.isnull().sum())  #display number of missing values in each column

# handling missing data
# dropna() method to remove rows with missing values
# print(df.dropna())  #display data with rows containing missing values removed

# fillna() method to fill missing values with a specified value
# print(df.fillna(0))  #display data with missing values filled with 0

# print(df['Age'].fillna(df['Age'].mean()))  # fill missing values in 'Age' column with mean of 'Age'

# print(df['Salary'].fillna(df['Salary'].mean()))  # fill missing values in 'salary' column with median of 'salary'


# interpolation method to fill missing values

# data={
#     'Age': [2, 30, None, 56, 94],
#     'Salary': [50000, 60000, 55000, None, 70000]
# }
# ddf=pd.DataFrame(data)
# print(ddf.interpolate(method='linear'))  #display data with missing values filled using interpolation 

# print(df['Age'].interpolate(method='linear', inplace=True)) # fill missing values in 'Age' column using interpolation

# sorting data
# data={
#     'name': ['John', 'Alice', 'Bob', 'Eve', 'Charlie'],
#     'Age': [2, 30, 23, 56, 94],
#     'Salary': [50000, 60000, 55000, 23000, 70000]
# }
# ddf=pd.DataFrame(data)
# ddf.sort_values(by='name' , ascending=[True, True], inplace=True)  # sort data by 'Name' and 'Age' columns in ascending order
# print(ddf)

# grouping  data
# data = {
#     "name": ["John", "Alice", "John", "Alice", "Bob"],
#     "Age": [25, 30, 25, 30, 25],
#     "Salary": [50000, 60000, 30000, 40000, 55000]
# }

# df = pd.DataFrame(data)

# print(df.groupby(["Age", "name"])["Salary"].sum())

# merging  data
# inner join
dt={
    'Employee_ID': [1001, 1002, 1003, 1004],
    'Name': ['John', 'Alice', 'Bob', 'Eve'],
    'Age': [25, 30, 23, 56]
}
df1=pd.DataFrame(dt)

dtt={
    'Employee_ID': [1001, 1002, 1005, 1006],
    'Salary': [50000, 60000, 55000, 70000]
}
df2=pd.DataFrame(dtt)

# # inner join
# merged_df=pd.merge(df1, df2, on='Employee_ID', how='inner')  # merge df1 and df2 on 'Employee_ID' column using inner join
# print(merged_df)

# # right join
# merged_df=pd.merge(df1, df2, on='Employee_ID', how='right')  # merge df1 and df2 on 'Employee_ID' column using right join
# print(merged_df)

# cross join
# merged_df=pd.merge(df1, df2, how='cross')  # merge df1
# print(merged_df)  #display merged data using cross join

# concatenating data   
  
  
