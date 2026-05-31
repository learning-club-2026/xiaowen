month=int(input("出生月份："))
day=int(input("出生日："))
if(month == 1 and 20<=day<=31) or (month == 2 and day <= 18):
        print ("水瓶座")
elif(month == 2 and 19<=day<=30) or (month == 3 and day <= 20):
        print("双鱼座")
elif(month == 3 and 21<=day<=31) or (month == 4 and day <= 19):
        print("白羊座")
elif (month == 4 and 20<=day<=30) or (month == 5 and day <= 20):
        print("金牛座")
elif (month == 5 and 21<=day<=31) or (month == 6 and day <= 21):
        print ("双子座")
elif (month == 6 and 22<=day<=30 ) or (month == 7 and day <= 22):
        print ("巨蟹座")
elif (month == 7 and 23<=day<=31) or (month == 8 and day <= 22):
        print ("狮子座")
elif (month == 8 and 23<=day<=31) or (month == 9 and day <= 22):
        print ("处女座")
elif (month == 9 and 23<=day<=30) or (month == 10 and day <= 23):
        print ("天秤座")
elif (month == 10 and 24<=day<=31) or (month == 11 and day <= 22):
        print("天蝎座")
elif (month == 11 and 23<=day<=30) or (month == 12 and day <= 21):
        print ("射手座")
elif (month == 12 and 22<=day<=31) or (month == 1 and day <= 19):
        print("摩羯座")
else:
        print(f"\n谁家孩子生日{month}月{day}日😅")