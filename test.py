lst = []
number = 1
for x in range(80):
    lst.append(number)
    number += 1


N = 10

for iteration in range(8):
    print(lst[0:N-10] + lst[N:])
    print(lst[N-10:N])
    N += 10