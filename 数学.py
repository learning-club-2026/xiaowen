num1=int(input("第一个百分比："))
num2=int(input("第二个百分比："))

n=1
while True:
    a = 0.01*num1 * n
    b = 0.01*num2 * a
    if a.is_integer() and b.is_integer():
        print(n)
        break
    n=n+1
