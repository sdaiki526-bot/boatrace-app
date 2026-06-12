from boatrace_scraper import BoatraceScraper
from datetime import date
sc = BoatraceScraper()
sc.login()
url = 'https://www.boatrace.jp/owpc/pc/race/racelist?hd=20260612&jcd=23&rno=1'
resp = sc.session.get(url)
open('racelist_0612.html', 'w', encoding='utf-8').write(resp.text)
print('保存完了')