import requests
from bs4 import BeautifulSoup

res = requests.get(url = "https://www.xinshipu.com/question")
print(res.text)

soup = BeautifulSoup(res.text,'html.parser')
target = soup.find(attrs={"class":"ask-list"})

li_list = target.find_all('li')

for item in li_list:
    #print(item)
    s = item.text.replace(" ", "").replace('\t', '')
    l = s.split('\n')
    print(l[0])