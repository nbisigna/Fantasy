from urllib.request import urlopen
from bs4 import BeautifulSoup
import csv

url = 'https://www.pro-football-reference.com/years/2022/fantasy.htm'

page = urlopen(url)

html_bytes = page.read()
html = html_bytes.decode('utf-8')

soup = BeautifulSoup(html, 'html.parser')

thead = ['\n', 'Rk', '\n', 'Player', '\n', 'Tm', '\n', 'FantPos', '\n', 'Age', '\n', 'G', '\n', 'GS', '\n', 'Cmp', '\n', 'Att', '\n', 'Yds', '\n', 'TD', '\n', 'Int', '\n', 'Att', '\n', 'Yds', '\n', 'Y/A', '\n', 'TD', '\n', 'Tgt', '\n', 'Rec', '\n', 'Yds', '\n', 'Y/R', '\n', 'TD', '\n', 'Fmb', '\n', 'FL', '\n', 'TD', '\n', '2PM', '\n', '2PP', '\n', 'FantPt', '\n', 'PPR', '\n', 'DKPt', '\n', 'FDPt', '\n', 'VBD', '\n', 'PosRank', '\n', 'OvRank', '\n']

for tr in soup.find_all('tr'):
    row = []
    if tr.has_attr('class'):
        continue
    for td in tr:
        text = td.text
        row.append(text)
    with open('football.csv', "a") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(row)