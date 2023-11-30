import requests
from bs4 import BeautifulSoup
import pandas as pd

def request_header_url(header, url_website):
    headers = {'User-Agent': header}
    return requests.get(url=url_website, headers=headers)

def parse_website(request):
    """Use html5lib library to parse"""
    return BeautifulSoup(request.content, 'html5lib')

def prettify_web_structure(parsed_page):
    parsed_page.prettify()

def get_content_table(web_element, tag, attributes):
    return web_element.find(tag, attrs=attributes)

def crawl_pta(url_link, id_prodi):
    # constant
    LAST_INDEX = -1
    FIRST_INDEX = 1
    INCREMENT_BY_ONE = 1

    r = request_header_url("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1", url_link)
    soup = parse_website(r)
    prettify_web_structure(soup)
    table = get_content_table(soup, "div", {"id":"wrapper"})

    pagination = table.findAll("a", attrs={"class":"pag_button"})
    total_pages = int(pagination[LAST_INDEX]["href"].split("/")[LAST_INDEX])

    papers = []
    for pages in range(FIRST_INDEX, total_pages + INCREMENT_BY_ONE):
        url_page = f"{url_link}/{pages}"

        r_pages = request_header_url("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36", url_page)
        soup_pages = parse_website(r_pages)
        prettify_web_structure(soup_pages)
        table_pages = get_content_table(soup_pages, "div", {"id":"wrapper"})

        for article_row in table_pages.findAll("li", attrs={"data-id":"id-1"}):
            journal_dict = {}
            title_journal = article_row.find('a', {'class':'title'}).text
            info_journal = article_row.findAll('span')
            writer_journal = info_journal[0].text.split(':')[1]
            supervisor1_journal = info_journal[1].text.split(':')[1]
            supervisor2_journal = info_journal[2].text.split(':')[1]
            url_journal = article_row.find('a', {'class':'gray button'}).get('href')
            req_journal = requests.get(url_journal)
            soup_journal = parse_website(req_journal)
            abstract_journal = soup_journal.find("p", attrs={"align":"justify"}).text

            journal_dict['Judul'] = title_journal
            journal_dict['Penulis'] = writer_journal
            journal_dict['Pembimbing 1'] = supervisor1_journal
            journal_dict['Pembimbing 2'] = supervisor2_journal
            journal_dict['Abstrak'] = abstract_journal
            journal_dict['Prodi'] = id_prodi

            papers.append(journal_dict)

    df_papers = pd.DataFrame(papers)
    df_papers.to_csv(f'PTA_{id_prodi}.csv', index=False)
