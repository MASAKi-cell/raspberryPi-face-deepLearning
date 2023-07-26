from flickrapi import FlickrAPI
from urllib.request import urlretrieve
from pprint import pprint
import os
import time
import sys

# APIキーの情報

key = "c9fca8f81f561723d5ac3c56de2c5ec9"
secret = "5c9e4f34b9632f76"
wait_time = 1

# 保存フォルダの指定
name = 'person'
savedir = "./" + name

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text=name,
    per_page=200,
    media='photos',
    sort='relevance',
    safe_search=1,
    extras='url_q, licence'
)

photos = result['photos']

for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath):
        continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)
