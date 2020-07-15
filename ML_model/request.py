import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Gender':0, 'Height(cm)':150, 'Weight(kg)':40})

print(r.json())