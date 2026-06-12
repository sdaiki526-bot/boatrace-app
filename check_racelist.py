from bs4 import BeautifulSoup
html = open('racelist_0612.html', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')
tables = soup.select('table')
target = max(tables, key=lambda t: len(t.select('tr')))
rows = target.select('tr')
print('総行数:', len(rows))
for i, row in enumerate(rows[3:7]):
    cells = row.select('td')
    print(f'行{i}: セル数={len(cells)}', [c.text.strip()[:15] for c in cells[:5]])