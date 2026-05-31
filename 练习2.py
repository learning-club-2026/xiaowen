import random


def simulate_coin_flips():
    """模拟抛硬币，直到出现连续3次正面(H)或反面(T)"""
    flips = []
    # 初始状态：没有连续相同的结果
    n = 1
    coins=['T', 'H']
    last_result=None

    while True:
        # 随机生成正面(H)或反面(T)，概率各50%
        result = (random.choice(coins))
        flips.append(result)

        # 检查是否与上一次结果相同
        if result == last_result:
            n += 1
            # 如果连续3次相同，结束循环
            if n == 3:
                break
        else:
            consecutive = 1  # 重置连续计数

        last_result = result

    # 返回投掷结果和总次数
    return ''.join(flips), len(flips)


# 执行10次模拟并计算平均值
total_flips = 0
num_simulations = 10

print("模拟结果：")
for i in range(num_simulations):
    sequence, count = simulate_coin_flips()
    print(f"{sequence} ({count} flips)")
    total_flips += count

# 计算并显示平均次数
average = total_flips / num_simulations
print(f"\n10次模拟的平均投掷次数：{average}")
