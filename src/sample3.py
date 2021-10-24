a = [[1], [2, 2], [3, 3, 3]]

b = [(array, len(array)) for array in a]
b.sort(key=lambda x: x[-1], reverse=True)
c = [array for array, size in b]
print(c)