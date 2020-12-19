#import libraries

import requests
import random
import time

#import data
from headers_data import ordered_headers_list

class Scraper():

    def __init__(self):
        self.headers_list = ordered_headers_list

    def select_header(self):
        self.header = random.choice(self.headers_list)

    def url_to_bytes_content(self, url):
        self.select_header()

        self.webpage_response = requests.get(url, headers = self.header)
        self.webpage_bytes = self.webpage_response.content

        return self.webpage_bytes

    def url_to_string_content(self, url):

        self.url_to_bytes_content(url)

        self.webpage_string = str(self.webpage_bytes)

        return self.webpage_string

    def webpage_string_to_html_regular(self, file_name):

        open(file_name+".html", "w").write(self.webpage_string)

    def sleep(self, max_sleep_time):
        time_list = range(0, max_sleep_time*100, 1)
        sleep_time = random.choice(time_list)/100
        time.sleep(sleep_time)

    def url_to_html_regular(self, url, file_name):
        self.url_to_string_content(url)
        self.webpage_string_to_html_regular(file_name)
