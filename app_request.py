import requests

url = 'https://peaceful-anchorage-79713.herokuapp.com/model'
myobj = {
            'age':31,
            'workclass':'Private',
            'fnlgt':45781,
            'education':'Masters',
            'education-num':14,
            'marital-status':'Never-married',
            'occupation':'Prof-specialty',
            'relationship':'Not-in-family',
            'race':'White',
            'sex':'Female',
            'capital-gain':14084,
            'capital-loss':0,
            'hours-per-week':50,
            'native-country':'United-States'
        }

resp = requests.post(url, json = myobj)

print(resp.status_code)
print(resp.json())