






import numpy as np

A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
b = np.array([0, 1, 2])
try:
    x = np.linalg.solve(A, b)
    print(x)
except np.linalg.LinAlgError:
    print("The given circuit is not solvable! Make sure that KCL and KVL can be obeyed at every node.")
    print("This may happen because of loops containing only voltage sources, nodes connected to only current sources")









# import re

# text_to_search = "abc Koushal IIT Madras Electrical Engineer 0123456789 ABC"

# pattern = re.compile('abc')

# def metricPrefixes2Value(value):
#     lastLetter = value[-1]
#     if lastLetter == 'k' or lastLetter == 'm' or lastLetter == 'u' or lastLetter == 'n' or lastLetter == 'p':
#         base = float(value[:-1])       # Everything other than the last prefix
#         Converter = { 'k': 1e3, 'm' : 1e-3, 'u' : 1e-6, 'n' : 1e-9, 'p' : 1e-12}
#         return base*Converter[lastLetter]
#     else:
#         return float(value)

# print(metricPrefixes2Value('4'))

# nodes_list = ['in3', 'GND', 'koushal', '3', '4']
# nodes_dictionary = {}
# i = 1
# for node in nodes_list:
#     if node != '0' and node!='GND':
#         nodes_dictionary[node] = i
#         i += 1
#     else:
#         nodes_dictionary[node] = 0

# print(nodes_dictionary)
# print(i)