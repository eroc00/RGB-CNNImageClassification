""" Classification categories
0) zero
1) one
2) two
3) three
4) four
5) five
6) confirm (thumbs up)
7) x
"""
label_num = 8

label_dict = {}
for i in range(label_num):
    if i == 6:
        label_dict[i] = "Thumbs Up"
    elif i == 7:
        label_dict[i] = "X"
    else:
        label_dict[i] = f"{i}"