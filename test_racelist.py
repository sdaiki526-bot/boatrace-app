from boatrace_scraper import BoatraceScraper
from crawler import get_holding_venues
from datetime import date

sc = BoatraceScraper()
sc.login()
venues = get_holding_venues(sc, date.today())
print('開催会場:', venues)

for venue in venues:
    for rno in range(1, 13):
        racers = sc.get_racelist(date.today(), venue, rno)
        if racers:
            from boatrace_scraper import VENUE_MAP
            print(f'取得成功: {VENUE_MAP[venue]} {rno}R 選手数:{len(racers)}')
            for r in racers:
                print(f'  {r.lane}枠 {r.racer_no} {r.name} {r.rank}')
            break