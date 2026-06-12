# test_result2.py として保存して実行
from boatrace_scraper import BoatraceScraper
from bs4 import BeautifulSoup

sc = BoatraceScraper()
html = open('test_result.html', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')

from datetime import date, timedelta
yesterday = date.today() - timedelta(days=1)
result = sc._parse_result(soup, yesterday, '03', 1)

print('着順:', result.arrival)
print('ST:', result.start_times)
print('払戻金:')
for k, v in result.payouts.items():
    print(f'  {k}: {v}')