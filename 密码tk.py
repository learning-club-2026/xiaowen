import tkinter as tk


def abc():
    password = int(entry.get())
    guess = 0
    while guess <= 99999999:
        if guess == password:
            result.config(text=f"破解密码为：{guess:08d}", fg="green")
            break
        guess += 1



root = tk.Tk()
root.title("密码破解")
root.geometry("400x300")


tk.Label(root, text="输入8位数密码:",font=("Arial", 15)).pack(pady=10)
entry = tk.Entry(root,show="*")
entry.pack(pady=5)


tk.Button(root, text="开始破解",font=("Arial", 15), command=abc).pack(pady=10)


result = tk.Label(root, text="",font=("Arial", 15))
result.pack(pady=10)

root.mainloop()
