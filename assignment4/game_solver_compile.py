# coding: utf-8
# pylint: disable=W0603
"""
@File        :   game_solver.py
@Time        :   2025/11/17 13:12:56
@Author      :   Usercyk
@Description :   Minimax
"""
from functools import lru_cache
import multiprocessing
import os
import cython

INF: cython.longlong = 10_000

row_count: cython.longlong = 0
cards = []
depth: cython.longlong = 0
first_move_row: cython.longlong = -1
first_move_side: cython.longlong = -1
dp = {}


def encode_state(state_list) -> cython.longlong:
    ret = 0
    for x in state_list:
        ret = ret * 100 + x
    return ret


def modify_state(state: cython.longlong, row: cython.longlong, column: cython.longlong, delta: cython.longlong) -> cython.longlong:
    pos = (row_count - 1 - row) * 4
    post = state % (10 ** pos)
    state = state // (10 ** pos)
    right = state % 100
    state = state // 100
    left = state % 100
    state = state // 100
    if left == right:
        return -1
    if right - left == 1:
        left = 0
        right = 0
    elif column == 0:
        left += delta
    else:
        right += delta
    state = state * 100 + left
    state = state * 100 + right
    state = state * (10 ** pos) + post
    return state


def get_left_index(state: cython.longlong, row: cython.longlong) -> cython.longlong:
    pos = (row_count - 1 - row) * 4
    state = state // (10 ** pos)
    state = state // 100
    left = state % 100
    return left


def get_right_index(state: cython.longlong, row: cython.longlong) -> cython.longlong:
    pos = (row_count - 1 - row) * 4
    state = state // (10 ** pos)
    right = state % 100
    return right


def gen_distribution(num: cython.longlong):
    if row_count == 4:
        return [[num0, num1, num2, num - num0 - num1 - num2] for num0 in range(
            max(num - len(cards[1]) - len(cards[2]) - len(cards[3]), 0),
            min(num + 1, len(cards[0]) + 1)) for num1 in range(
                max(num - num0 - len(cards[2]) - len(cards[3]), 0),
                min(num - num0 + 1, len(cards[1]) + 1)) for num2 in range(
                    max(0, num - num0 - num1 - len(cards[3])),
                    min(num - num0 - num1 + 1, len(cards[2]) + 1))]
    if row_count == 3:
        return [[num0, num1, num - num0 - num1] for num0 in range(
            max(num - len(cards[1]) - len(cards[2]), 0),
            min(num + 1, len(cards[0]) + 1)) for num1 in range(
                max(0, num - num0 - len(cards[2])),
                min(num - num0 + 1, len(cards[1]) + 1))]
    if row_count == 2:
        return [[num0, num - num0]
                for num0 in range(
            max(0, num - len(cards[1])),
            min(num + 1, len(cards[0]) + 1))]

    return [[num]]


def get_initial_state_list(dist):
    if row_count == 4:
        return [encode_state([start_0, start_0+dist[0], start_1,
                              start_1+dist[1], start_2, start_2+dist[2],
                              start_3, start_3+dist[3]])
                for start_0 in range(0, 1 if dist[0] == 0 else len(cards[0]) - dist[0] + 1)
                for start_1 in range(0, 1 if dist[1] == 0 else len(cards[1]) - dist[1] + 1)
                for start_2 in range(0, 1 if dist[2] == 0 else len(cards[2]) - dist[2] + 1)
                for start_3 in range(0, 1 if dist[3] == 0 else len(cards[3]) - dist[3] + 1)]
    if row_count == 3:
        return [encode_state([start_0, start_0+dist[0],
                              start_1, start_1+dist[1],
                              start_2, start_2+dist[2]])
                for start_0 in range(0, 1 if dist[0] == 0 else len(cards[0]) - dist[0] + 1)
                for start_1 in range(0, 1 if dist[1] == 0 else len(cards[1]) - dist[1] + 1)
                for start_2 in range(0, 1 if dist[2] == 0 else len(cards[2]) - dist[2] + 1)]
    if row_count == 2:
        return [encode_state([start_0, start_0+dist[0], start_1, start_1+dist[1]])
                for start_0 in range(0, 1 if dist[0] == 0 else len(cards[0]) - dist[0] + 1)
                for start_1 in range(0, 1 if dist[1] == 0 else len(cards[1]) - dist[1] + 1)]

    return [encode_state([start_0, start_0+dist[0]])
            for start_0 in range(0, 1 if dist[0] == 0 else len(cards[0]) - dist[0] + 1)]


def task(dist):
    ret_dp = {}
    state_list = get_initial_state_list(dist)
    for state in state_list:
        ret_dp[state] = -INF
        for optionrow in range(row_count):
            if dist[optionrow] == 0:
                continue

            newstate = modify_state(state, optionrow, 0, 1)
            score_gain = cards[optionrow][get_left_index(
                state, optionrow)]
            newscore = -dp[newstate] + score_gain
            if newscore > ret_dp[state]:
                ret_dp[state] = newscore

            if dist[optionrow] == 1:
                continue
            newstate = modify_state(state, optionrow, 1, -1)
            score_gain = cards[optionrow][get_right_index(
                state, optionrow) - 1]
            newscore = -dp[newstate] + score_gain
            if newscore > ret_dp[state]:
                ret_dp[state] = newscore
    return ret_dp


@lru_cache(maxsize=None)
def minimax(state: cython.longlong) -> cython.longlong:
    global depth, first_move_row, first_move_side

    if state in dp:
        return dp[state]

    best: cython.longlong = -INF

    for row in range(row_count):
        lft = get_left_index(state, row)
        rgt = get_right_index(state, row)
        if lft == rgt:
            continue

        nxtstate = modify_state(state, row, 0, 1)
        depth += 1
        nxtscore = minimax(nxtstate)
        depth -= 1
        nxtscore = -nxtscore + cards[row][lft]
        if nxtscore > best:
            best = nxtscore
            if not depth:
                first_move_row = row
                first_move_side = 0

        if rgt - lft == 1:
            continue

        nxtstate = modify_state(state, row, 1, -1)
        depth += 1
        nxtscore = minimax(nxtstate)
        depth -= 1
        nxtscore = -nxtscore + cards[row][rgt - 1]
        if nxtscore > best:
            best = nxtscore
            if not depth:
                first_move_row = row
                first_move_side = 1

    return best


def run() -> None:
    global row_count, cards, dp
    with open("input.txt", "r", encoding="utf-8") as fin:
        n: cython.longlong = int(fin.readline().strip())
        cards = [
            list(map(int, line.strip().split(",")))
            for line in fin.readlines()[:n]
        ]
        row_count: cython.longlong = len(cards)

    total_val: cython.longlong = n*(n+1)//2

    initial_state_list = []
    for i in range(row_count):
        initial_state_list.append(0)
        initial_state_list.append(len(cards[i]))
    initial_state: cython.longlong = encode_state(initial_state_list)

    dp[0] = 0
    num_processes = os.cpu_count()
    assert num_processes is not None
    formaxn = min(n - 1, int(n - n // 2.8))
    for num in range(1, formaxn):
        dists = gen_distribution(num)
        if len(dists) > num_processes * 0.3:
            with multiprocessing.Pool(processes=min(num_processes, len(dists))) as pool:
                results = pool.map(task, dists)
            for ret_dp in results:
                dp |= ret_dp
        else:
            for numm in dists:
                dp |= task(numm)

    val: cython.longlong = minimax(initial_state)
    side = "右端" if first_move_side else "左端"
    card_val = cards[first_move_row][-1 if first_move_side else 0]
    with open("output.txt", "w", encoding="utf-8") as fout:
        fout.write(f"第{first_move_row+1}行 {side} 牌点数{card_val}\n")
        fout.write(f"小红: {(total_val+val)//2} 小蓝: {(total_val-val)//2}")
