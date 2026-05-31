from datetime import *
a=date(2007,1,3)
b=date.today()
num=0
while a<b:
    if a.month==1 and a.day==3 and a.weekday()==6:
        num+=1
        print(a)
    a+=timedelta(1)

print(num)