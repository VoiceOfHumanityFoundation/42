# -*- coding: utf-8 -*-
import requests
import json
import chardet
import sys
#data = b"\x95\xc3\x8a\xb0\x8ds\x86\x89\x94\x82\x8a\xba"
#detected = chardet.detect(data)
#decoded = data.decode(detected["encoding"])

request_example = {"message" : sys.argv[1]}
#get_test_url= f"http://127.0.0.1:8000/chunker"
get_test_url= f"http://127.0.0.1:8000/rag"
#r = requests.post(get_test_url, json=request_example).decode('unicode_escape')
r = requests.post(get_test_url, json=request_example)
counter = 1
for elem in r:
    print(elem.decode('unicode_escape'))
    #print(elem.decode('unicode_escape').encode("latin-1").decode('unicode_escape'))
    #print(elem)
    #print(counter)
    #counter += 1

