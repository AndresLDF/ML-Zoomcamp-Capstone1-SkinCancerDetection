import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://media-cldnry.s-nbcnews.com/image/upload/t_fit-560w,f_auto,q_auto:eco,dpr_2.0/rockcms/2023-08/melanoma-pictures-mc-230804-02-398f58.jpg'}

result = requests.post(url, json=data).json()
print(result)