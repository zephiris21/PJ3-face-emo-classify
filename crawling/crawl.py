# 설치: pip install icrawler
from icrawler.builtin import GoogleImageCrawler


keyword = "운명"
# 기본 사용법
google_crawler = GoogleImageCrawler(storage={'root_dir': keyword})
google_crawler.crawl(keyword=keyword, max_num=50)

# # 고급 필터링 옵션
# filters = dict(
#     size='large',           # 큰 이미지만
#     # color='',               # 색상 필터 (필요시)
#     # license='commercial',   # 라이센스 유형
#     # date=((2023, 1, 1), (2025, 5, 20))  # 날짜 범위
# )
# google_crawler.crawl(keyword=keyword, filters=filters, max_num=100)