try:
    a = int(input())
except ValueError:
    print("the input is not numerical")
else:

    if a >= 90:
        print("A")
    elif 80 <= a < 90:
        print("B")
    elif 70 <= a < 80:
        print("C")
    elif 60 <= a < 70:
        print("D")
    elif a < 60:
        print("E")
    else:
        print("other error")

