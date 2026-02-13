# LU Decomposition 

## AIM:
To write a program to find the LU Decomposition of a matrix.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the NumPy library.
2. Define the given square matrix using np.array().
3. Use scipy.linalg.lu() to decompose the matrix into Lower triangular (L) and Upper triangular (U) matrices.
4. Display the L and U matrices and verify that 
𝐴
=
𝐿
𝑈
A=LU.

## Program:
(i) To find the L and U matrix
```
'''Program to find L and U matrix using LU decomposition.
Developed by: Mukesh R
RegisterNumber: 212224240098
'''
import numpy as np

A = np.array(eval(input()), dtype=float)
n = len(A)

L = np.zeros((n, n))
U = A.copy()

for i in range(n):
    max_row = np.argmax(abs(U[i:, i])) + i
    if max_row != i:
        U[[i, max_row]] = U[[max_row, i]]
        L[[i, max_row], :i] = L[[max_row, i], :i]

    L[i][i] = 1.0

    for j in range(i + 1, n):
        factor = U[j][i] / U[i][i]
        L[j][i] = factor
        U[j] = U[j] - factor * U[i]

print(L)
print(U)
```
(ii) To find the LU Decomposition of a matrix
```
'''Program to solve a matrix using LU decomposition.
Developed by: Mukesh R
RegisterNumber: 212224240098
'''

import numpy as np

A = np.array(eval(input()), dtype=float)
B = np.array(eval(input()), dtype=float)

n = len(A)

L = np.zeros((n, n))
U = A.copy()

for i in range(n):
    max_row = np.argmax(abs(U[i:, i])) + i
    if max_row != i:
        U[[i, max_row]] = U[[max_row, i]]
        B[[i, max_row]] = B[[max_row, i]]
        L[[i, max_row], :i] = L[[max_row, i], :i]

    L[i][i] = 1.0

    for j in range(i + 1, n):
        factor = U[j][i] / U[i][i]
        L[j][i] = factor
        U[j] = U[j] - factor * U[i]

Y = np.zeros(n)
for i in range(n):
    Y[i] = B[i] - np.dot(L[i, :i], Y[:i])

X = np.zeros(n)
for i in range(n - 1, -1, -1):
    X[i] = (Y[i] - np.dot(U[i, i+1:], X[i+1:])) / U[i][i]
print(X)
```

## Output:

<img width="1183" height="438" alt="image" src="https://github.com/user-attachments/assets/03b81624-780a-44e3-9e77-adf5fd54bcd9" />

<img width="929" height="183" alt="image" src="https://github.com/user-attachments/assets/847f5e6e-29cd-43b7-a050-094659bdf8c7" />


## Result:
Thus the program to find the LU Decomposition of a matrix is written and verified using python programming.

