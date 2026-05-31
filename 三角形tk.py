import tkinter as tk


def check_triangle():
    a = int(a_entry.get())
    b = int(b_entry.get())
    c = int(c_entry.get())


    if a <= 0 or b <= 0 or c <= 0:
        result.config(text="错误，边长需大于0", fg="red")
        return

    lengths = [a, b, c]
    min_len = min(lengths)
    max_len = max(lengths)
    mid_len = a + b + c - max_len - min_len


    if mid_len + min_len > max_len:
        result.config(text="可以构成三角形", fg="green")
    else:
         result.config(text="不能构成三角形", fg="red")


root = tk.Tk()
root.title("三角形")
root.geometry("300x400")


tk.Label(root, text="第一根长度:",font=("Arial", 15)).pack(pady=5)
a_entry = tk.Entry(root)
a_entry.pack(pady=5)

tk.Label(root, text="第二根长度:",font=("Arial", 15)).pack(pady=5)
b_entry = tk.Entry(root)
b_entry.pack(pady=5)

tk.Label(root, text="第三根长度:",font=("Arial", 15)).pack(pady=5)
c_entry = tk.Entry(root)
c_entry.pack(pady=5)


tk.Button(root, text="判断",font=("Arial", 15), command=check_triangle).pack(pady=10)


result = tk.Label(root, text="", font=("Arial", 20))
result.pack(pady=20)

root.mainloop()
