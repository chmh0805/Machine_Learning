
N = int(input())
for _ in range(N) :
    test = list(input())
    score = [0] * len(test)
    if test[0] == "O" :
        score[0] = 1
    for i in range (1, len(test)) :
        if test[i-1] == test[i] == "O" :
            score[i] = score[i-1] + 1
        elif test[i] == "O" :
            score[i] = 1
    print(sum(score))