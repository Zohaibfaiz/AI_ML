import numpy as np

#average
# array=[1, 2, 3, 4, 5]
# average = np.mean(array)
# print("The average is:", average)

# ARRAY CREATION
# lst=[1, 2, 3, 4, 5]
# arr=np.array(lst)
# print(type(arr))

#np.zeros
# arr=np.zeros((5,2))
# print(arr)

#np.ones
# arr=np.ones((5))
# print(arr)


# np.arange like range function
# arr=np.arange(1, 10, 2)
# print(arr)

# np.linspace
# arr=np.linspace(1, 10, 5)
# print(arr)

# temp =np.linspace(20,40,4)
# print(temp)

# ary=np.zeros(8)
# print(ary)

# arry=np.zeros(2,5)
# print(arry)

# arry=np.arange(10,51)
# print(arry)

# arr=np.arange(0,100,10)
# print(arr)

# arr=np.linspace(0,1,7)
# print(arr)

# 3d
# a = np.array([
#     [
#         [1,2],
#         [3,4]
#     ],

#     [
#         [5,6],
#         [7,8]
#     ]
# ])

# print(a.ndim)
# print(a.shape)
# print(a.size)


# arr = np.array([[10,20,30],[40,50,60]])
# print(arr[2:0:-1]) slice
# print(arr[1,1]) indexing
# print(arr.ndim)


# arr = np.array([10, 20, 30, 40, 50])
# print(arr[4:0:-1])
  
# filtering
# arr = np.array([10, 20, 30, 40, 50])
# print(arr[arr < 30])

# marks = np.array([35,55,67,82,91,48])

# print(marks[(marks >= 50) & (marks <= 80)])
  
  
# reshaping
# arr = np.array([1, 2, 3,4, 5, 6,7,8,9,10,11,12])
# print(arr.reshape(6,2))

# arr = np.array([
#     [5,10,15],
#     [20,25,30],
#     [35,40,45]
# ])
# print(arr[1:,2:])

# ravel and flatten
# import numpy as np

# arr = np.array([
#     [1, 2, 3],
#     [4, 5, 6]
# ])

# # print(arr.ravel())
# # print(arr.flatten())
# naw=arr.flatten()
# naw[0]=100
# print(naw)
# print(arr)

# new=arr.ravel()
# new[1]=100
# print(arr)
# print(new)


# broadcasting
# marks = np.array([50, 60, 70, 80, 90])
# print(marks +5)

# arr = np.array([2,4,6])
# print(arr * 3)

# a = np.array([1,2,3])

# b = np.array([10,20,30])

# print(a + b)  # Element-wise addition

# a = np.array([
#     [1,2,3],
#     [4,5,6]
# ])

# b = np.array([10,20,30])
# print(a + b)  # Broadcasting: adds b to each row of a

# a = np.array(
#     [1,2,3])

# b = np.array([[10,20,30],[1,2,3]])
# print(a + b)  # Broadcasting: adds b to each row of a

# aggregate functions
# np.sum()

# np.mean()

# np.median()

# np.max()

# np.min()

# np.std()

# np.var()

# arr = np.array([10,20,30])
# arr=np.append(arr, [40,50])
# print(arr)

# arr = np.array([10,20,30])
# arr=np.insert(arr, 1, [15])
# print(arr)

# arr = np.array([10,20,30])
# arr=np.delete(arr, 1)
# print(arr)

# searching
# arr = np.array([10,20,30,40])

# print(np.where(arr==30))

# unique
# arr = np.array([1,2,2,3,3,3])

# print(np.unique(arr))

# concatenation
# arr1 = np.array([1,2,3])
# arr2 = np.array([4,5,6])
# arr3 = np.concatenate([arr1, arr2])
# print(arr3)

# arr2= np.array([[1,2,3],[4,5,6]])
# new_arr = np.delete(arr2, 1, axis=0)  
# # Delete the second row
# print(new_arr)


# vstack()
# hstack()
# arr1 = np.array([1,2,3])
# arr2 = np.array([4,5,6])

# print(np.vstack((arr1, arr2)))  # Stack vertically
# print(np.hstack((arr1, arr2)))  # Stack horizontally

#  splitting
# arr = np.array([1,2,3,4,5,6])
# arr1, arr2 = np.split(arr, 2)  # Split into 2 equal parts
# print(arr1)
# print(arr2)

# hsplit
# vsplit
# arr2 = np.array([[1,2,3],[4,5,6]])
# arr1, arr2 = np.vsplit(arr2, 2)  # Split vertically
# print(arr1)
# print(arr2)

# arr2 = np.array([[1,2,3,2],[2,4,5,6]])
# arr1, arr2 = np.hsplit(arr2, 2)  # Split horizontally
# print(arr1)
# print(arr2)

# # parts
# import numpy as np

# arr = np.array([
#     [1,2,3,4],
#     [5,6,7,8]
# ])

# parts = np.hsplit(arr, 2)

# print("First Part:")
# print(parts[0])

# print()

# print("Second Part:")
# print(parts[1])

# p=np.array([100,200,300,400,500])
# dis=10
# fp=p-(dis*p/100)
# print(fp)

# arr_1d=np.array([1,2])
# arr3=np.reshape(arr_1d,()) #add value at [0,3] index
# print(arr3)


# arr_2d=np.array([[1,2,3],[4,5,6]])
# result=arr_2d+arr_1d
# print(result)

# vectorization
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6])
# result = arr1 * arr2
# print(result)  # Output: [4 10 18] 


# find missing value in array by np.nan
# arr=np.array([1, 2, np.nan, 4, 5])
# missing_values = np.isnan(arr)
# print("Missing values:", missing_values) 

# arr = np.nan_to_num(arr, nan=0)  # Replace NaN with 0
# print("Array after replacing NaN with 0:", arr)


# handle infinity
# arr = np.array([1, 2, np.inf, 4, -np.inf])
# finite_values = np.isfinite(arr)
# print("Finite values:", finite_values)  # Output: [ True  True False  True False] 

# # initialize infinity values to a specific number
# arr[np.isinf(arr)] = 0  # Replace infinity with 0
# clean=np.nan_to_num(arr, posinf=0, neginf=0)  # Replace positive and negative infinity with 0
# print("Array after replacing infinity with 0:", arr)  # Output: [1. 2. 0. 4. 0.]

