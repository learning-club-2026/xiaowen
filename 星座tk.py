import tkinter as tk


def abc():
        month = int(month_entry.get())
        day = int(day_entry.get())

        if (month == 1 and 20 <= day <= 31) or (month == 2 and day <= 18):
            result.config(text="水瓶座")
        elif (month == 2 and 19 <= day <= 29) or (month == 3 and day <= 20):
            result.config(text="双鱼座")
        elif (month == 3 and 21 <= day <= 31) or (month == 4 and day <= 19):
            result.config(text="白羊座")
        elif (month == 4 and 20 <= day <= 30) or (month == 5 and day <= 20):
            result.config(text="金牛座")
        elif (month == 5 and 21 <= day <= 31) or (month == 6 and day <= 21):
            result.config(text="双子座")
        elif (month == 6 and 22 <= day <= 30) or (month == 7 and day <= 22):
            result.config(text="巨蟹座")
        elif (month == 7 and 23 <= day <= 31) or (month == 8 and day <= 22):
            result.config(text="狮子座")
        elif (month == 8 and 23 <= day <= 31) or (month == 9 and day <= 22):
            result.config(text="处女座")
        elif (month == 9 and 23 <= day <= 30) or (month == 10 and day <= 23):
            result.config(text="天秤座")
        elif (month == 10 and 24 <= day <= 31) or (month == 11 and day <= 22):
            result.config(text="天蝎座")
        elif (month == 11 and 23 <= day <= 30) or (month == 12 and day <= 21):
            result.config(text="射手座")
        elif (month == 12 and 22 <= day <= 31) or (month == 1 and day <= 19):
            result.config(text="摩羯座")
        else:
            result.config(text="请输入正确日期")

root = tk.Tk()
root.title("星座查询")
root.geometry("500x300")

# 月份
tk.Label(root, text="出生月份：", font=("Arial", 15)).pack(pady=5)
month_entry = tk.Entry(root)
month_entry.pack(pady=5)

# 日
tk.Label(root, text="出生日：", font=("Arial", 15)).pack(pady=5)
day_entry = tk.Entry(root)
day_entry.pack(pady=5)

tk.Button(root, text="查询",  font=("Arial", 15), command=abc).pack(pady=15)


result = tk.Label(root, text="", font=("Arial", 20), fg="green")
result.pack(pady=20)

root.mainloop()
