from icrawler.builtin import GoogleImageCrawler

def download_from_google(query, folder, num_images=100):
    crawler = GoogleImageCrawler(storage={'root_dir': folder})
    crawler.crawl(keyword=query, max_num=num_images)

# Ejemplo:
download_from_google("laptop", "dataset/laptop", 500)
download_from_google("smartphone", "dataset/celular", 500)
download_from_google("imagenes de tablets fondo blanco", "dataset/tablet", 500)
