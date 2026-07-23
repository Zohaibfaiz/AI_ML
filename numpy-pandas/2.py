import pandas as pd

# df=pd.read_csv('employee data.csv' )
# print(df.head())


# create data for csv
data ={
    'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
    'Age': [27, 24, 22, 32],
    'city' : ['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
    'salary':[10000, 20000, 30000, 40000]
}

df=pd.DataFrame(data)
# print(df)

# df.to_csv('employee data.csv', index=False)  #inde
# df.to_excel('employee data.xlsx', index=False)  #index=False means do not write row numbers


# read csv file
# df=pd.read_csv('Employee data.csv' )
# print(df)

# reading from other folder
# df=pd.read_csv('other_folder/employee data.csv')

# head dispaly specified number of rows
# print(df.head(2))  #display first 2 rows

# print(df.tail(2))  #display last 2 rows

# shape of data
# print(df.shape)  #display number of rows and columns

# print(df.shape[0]) #display number of rows

# print(df.columns)   display column names

# index
# print(df.index)  #display row indices

# datatype of each column
# print(df.dtypes)  #display datatype of each column

# info about data
# print(df.info())  #display info about data

# describe
# print(df.describe())  #display statistical summary of numerical columns

# conditional selection
# print(df[df['Age']>25])  #display rows where Age is greater than

# updating values
# df.loc[df['Age'] > 25, 'Age'] = 30  # update Age to 30 where it is greater than 25
# print(df)

# bonus addition in salary
# df['bonus']=df['salary']*0.1  # add a new column 'bonus' which is 10% of salary
# print(df)


# using insert method to add a new column at a specific position
# df.insert(loc=0, column='Employee_ID', value=[1001, 1002, 1003, 1004])  # insert 'Employee_ID' column at index 3
# print(df)


# pd.set_option("display.max_columns", None)

# update at specific index
# df.loc[0, 'salary'] = 30000  # update salary of employee at index 3 to 50000
# print(df)

# loc used to access a group of rows and columns by labels or a boolean array

# drop method to remove rows or columns
# df.drop(columns=['Age'],  inplace=True)  # drop the 'Age' column
# print(df)

# handle mising data
# isnull() method to check for missing values