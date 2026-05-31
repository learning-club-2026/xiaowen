import math
name=input()
time=int(input())
hour=int(input())
shuilv=float(input())
bili=float(input())
income=hour*shuilv
shuifei=income*shuilv
yanglao=income*bili
jin1=income-shuifei-yanglao
jin=round(jin1,2)
kouchu1=income-jin1
kouchu=round(kouchu1,2)
print(f"Employee Name: {name}\n\n")
print(f"Hours Worked:  {time}")
print(f"Pay Rate: ${hour}")
print(f"Gross Pay: ${income}")
print(f"Deductions: {kouchu:.2f}")
print(f"Net Pay: {jin:.2f}")