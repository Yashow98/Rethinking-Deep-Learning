# @Author  : YashowHoo
# @File    : shallow_deep_copy.py
# @Description :
import copy

l = [0, 1, [2, 3]]   # contain mutable objects like lists or dicts, shallow copy will create references to the original objects
l_assign = l                   # assignment
l_copy = l.copy()              # shallow copy
l_deepcopy = copy.deepcopy(l)  # deep copy

l[1] = 100
l[2][0] = 200
print(l)
# [0, 100, [200, 3]]

print(l_assign)
# [0, 100, [200, 3]]

print(l_copy)
# [0, 1, [200, 3]]

print(l_deepcopy)
# [0, 1, [2, 3]]

L1 = [1, 2, 3]
L2 = L1.copy()
print(L1 is L2)

L2[0] = 0
print(L1, L2)
