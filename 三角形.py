a=int(input("第一根长度:"))
b=int(input("第二根长度:"))
c=int(input("第三根长度:"))
def h(a,b,c):
  if a <= 0 or b <= 0 or c <= 0:
      prin
      return False

if h(a,b,c)==False:
    exit()

length = [a,b,c]
min_len=min(length)
max_len=max(length)
mid_len=(a+b+c - max_len - min_len)
if mid_len + min_len > max_len :
    print("\n可以构成三角形")
else:
    print("\n不能构成三角形")