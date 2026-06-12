from bs4 import BeautifulSoup
html = open('test_result.html', encoding='utf-8').read()
soup = BeautifulSoup(html, 'html.parser')
tables = soup.select("table.is-w495")
print("3つ目のテーブル tbody数:", len(tables[2].select("tbody")))
for i, tbody in enumerate(tables[2].select("tbody")[:4]):
    print(f"\ntbody[{i}]:")
    for row in tbody.select("tr"):
        cells = row.select("td")
        print("  rowspan:", cells[0].get("rowspan") if cells else "-")
        print("  texts:", [c.text.strip()[:30] for c in cells[:4]])