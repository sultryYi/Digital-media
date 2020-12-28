def mul(a, b):
    point_a = a.find('.')
    point_b = b.find('.')
    len_a = len(a)
    len_b = len(b)
    if point_a == -1:
        move_a = 0
    else:
        move_a = len_a - 1 - point_a
    if point_b == -1:
        move_b = 0
    else:
        move_b = len_b - 1 - point_b
    move = move_a + move_b

    split_a = ''
    split_b = ''
    for i in a.split('.'):
        split_a = split_a + i
    for i in b.split('.'):
        split_b = split_b + i

    temp_a = int(split_a)
    temp_b = int(split_b)

    preRes = str(temp_a * temp_b)
    res = preRes
    if move != 0:
        if move >= len(preRes):
            preRes = preRes.zfill(move + 1)
        res = preRes[0:-move] + '.' + preRes[-move:]

    return res


def add(a, b):
    point_a = a.find('.')
    point_b = b.find('.')
    len_a = len(a)
    len_b = len(b)
    if point_a == -1:
        move_a = 0
    else:
        move_a = len_a - 1 - point_a
    if point_b == -1:
        move_b = 0
    else:
        move_b = len_b - 1 - point_b

    move = max(move_a, move_b)

    split_a = ''
    split_b = ''
    for i in a.split('.'):
        split_a = split_a + i
    for i in b.split('.'):
        split_b = split_b + i

    if move_a > move_b:
        split_b = split_b + '0' * (move_a - move_b)
    elif move_a < move_b:
        split_a = split_a + '0' * (move_b - move_a)

    preRes = str(int(split_a) + int(split_b))
    res = preRes
    if move != 0:
        if move >= len(preRes):
            preRes = preRes.zfill(move + 1)
        res = preRes[0:-move] + '.' + preRes[-move:]
    return res


def div(a, b):
    point_a = a.find('.')
    point_b = b.find('.')
    len_a = len(a)
    len_b = len(b)
    if point_a == -1:
        move_a = 0
    else:
        move_a = len_a - 1 - point_a
    if point_b == -1:
        move_b = 0
    else:
        move_b = len_b - 1 - point_b

    move = max(move_a, move_b)

    split_a = ''
    split_b = ''
    for i in a.split('.'):
        split_a = split_a + i
    for i in b.split('.'):
        split_b = split_b + i

    if move_a > move_b:
        split_b = split_b + '0' * (move_a - move_b)
    elif move_a < move_b:
        split_a = split_a + '0' * (move_b - move_a)

    preRes = str(int(split_a) - int(split_b))
    res = preRes
    if move != 0:
        if move >= len(preRes):
            preRes = preRes.zfill(move + 1)
        res = preRes[0:-move] + '.' + preRes[-move:]
    return res


def compare(a, b):
    point_a = a.find('.')
    point_b = b.find('.')
    len_a = len(a)
    len_b = len(b)
    if point_a == -1:
        move_a = 0
    else:
        move_a = len_a - 1 - point_a
    if point_b == -1:
        move_b = 0
    else:
        move_b = len_b - 1 - point_b

    move = max(move_a, move_b)

    split_a = ''
    split_b = ''
    for i in a.split('.'):
        split_a = split_a + i
    for i in b.split('.'):
        split_b = split_b + i

    if move_a > move_b:
        split_b = split_b + '0' * (move_a - move_b)
    elif move_a < move_b:
        split_a = split_a + '0' * (move_b - move_a)

    preRes = int(split_a) - int(split_b)

    if preRes > 0:
        return 1
    elif preRes == 0:
        return 0
    else:
        return -1