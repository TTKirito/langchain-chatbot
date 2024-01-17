import schedule
import time

from craw.craw import craw, embeddingCraw

def job():
    print("I'm working...")
    crawl_df = craw('https://vinova.sg')
    embeddingCraw(crawl_df)
    print("OK")

schedule.every(1).minutes.do(job)

if __name__ == '__main__':
    while 1:
        schedule.run_pending()
        time.sleep(1)