scores = {'mike': 77, 'jack': 11}

len(scores)
#  通过key 访问value 字典名[key] = value
var = scores['mike']

x = scores.get('mike')
scores['ike'] = 0
del scores['ike']
print(var)
print(x)
