import tkinter as tk
from tkinter import messagebox


def abc():
    try:
        month = int(month())
        day = int(day())

        if (month == 1 and 20 <= day <= 31) or (month == 2 and day <= 18):
            result_label.config(text="水瓶座")
        elif (month == 2 and 19 <= day <= 29) or (month == 3 and day <= 20):
            result_label.config(text="双鱼座")
        elif (month == 3 and 21 <= day <= 31) or (month == 4 and day <= 19):
            result_label.config(text="白羊座")
        elif (month == 4 and 20 <= day <= 30) or (month == 5 and day <= 20):
            result_label.config(text="金牛座")
        elif (month == 5 and 21 <= day <= 31) or (month == 6 and day <= 21):
            result_label.config(text="双子座")
        elif (month == 6 and 22 <= day <= 30) or (month == 7 and day <= 22):
            result_label.config(text="巨蟹座")
        elif (month == 7 and 23 <= day <= 31) or (month == 8 and day <= 22):
            result_label.config(text="狮子座")
        elif (month == 8 and 23 <= day <= 31) or (month == 9 and day <= 22):
            result_label.config(text="处女座")
        elif (month == 9 and 23 <= day <= 30) or (month == 10 and day <= 23):
            result_label.config(text="天秤座")
        elif (month == 10 and 24 <= day <= 31) or (month == 11 and day <= 22):
            result_label.config(text="天蝎座")
        elif (month == 11 and 23 <= day <= 30) or (month == 12 and day <= 21):
            result_label.config(text="射手座")
        elif (month == 12 and 22 <= day <= 31) or (month == 1 and day <= 19):
            result_label.config(text="摩羯座")
        else:
            result_label.config(text="输入错误")
    except ValueError:
        messagebox.showerror("错误", "请输入数字")


root = tk.Tk()
root.title("星座查询")
root.geometry("250x200")


tk.Label(root, text="出生月份：").pack(pady=5)
month_entry.pack(pady=5)


tk.Label(root, text="出生日：").pack(pady=5)
day_entry = tk.Entry(root)
day_entry.pack(pady=5)

tk.Button(root, text="查询", command=get_zodiac).pack(pady=10)

result_label = tk.Label(root, text="",bg="silver", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()