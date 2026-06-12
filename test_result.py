from boatrace_scraper import BoatraceScraper
from datetime import date, timedelta
sc = BoatraceScraper()
sc.login()
yesterday = date.today() - timedelta(days=1)
url = 'https://www.boatrace.jp/owpc/pc/race/raceresult?hd=' + yesterday.strftime('%Y%m%d') + '&jcd=03&rno=1'
resp = sc.session.get(url)
print('データなし:', 'データがありません' in resp.text)
open('test_result.html', 'w', encoding='utf-8').write(resp.text)
print('保存完了')