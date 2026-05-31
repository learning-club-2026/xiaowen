import random
def simulate():
    flips = []
    coins = ['T', 'H']
    while True:

        result = random.choice(coins)
        flips.append(result)
        if len(flips)>=3  and  flips[-1]==flips[-2]==flips[-3]:
            break

    return '\t'.join(flips), len(flips)

n = 0
for a in range(10):
    abc, count = simulate()
    print(f"{abc} \t({count} flips)")
    n += count
print(f"\nOn average, {n / 10} flips were needed. ")
