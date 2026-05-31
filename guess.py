m=eval(input())
n=eval(input())
if m<n:
    m,n=n,m
while n!=0:
    a=m%n
    m=n
    n=a
print(m)
