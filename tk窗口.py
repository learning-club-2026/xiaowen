import tkinter as tk

root = tk.Tk()
root.title('坤坤')
root.geometry('600x500')

var = tk.StringVar()

l=tk.Label(root,textvariable=var,bg='silver',font=('Arial',20),width=1000, height=2)
l.pack()

on_hit=False
def hit_me():
    global on_hit
    if on_hit == False:
       on_hit=True
       var.set('你干嘛，嗨害，哎呦～')
    else:
       on_hit=False
       var.set('小黑子食不食油饼')

b=tk.Button(root,text='猛击魔丸',font=('Arial',20),width=100,height=2,command=hit_me)
b.pack()

root.mainloop()
