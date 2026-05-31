passward=int(input("输入8位数密码:"))
guess=0
while guess<=99999999:
    if guess == passward:
        print(f"破解密码为：{guess:08d}")
        break
    guess+=1

