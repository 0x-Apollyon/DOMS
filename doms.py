import requests
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
from bs4 import UnicodeDammit
from langdetect import detect
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import time
from skimage.metrics import structural_similarity as ssim
import os

parse = argparse.ArgumentParser()

parse.add_argument('-w1','--website_1',help="URL of first website",required=True)
parse.add_argument('-w2','--website_2',help="URL of second website",required=True)
parse = parse.parse_args()

if os.name == "nt":
    os.system("cls")
else:
    os.system("clear")
print("Running tests ...")

def url_safety(url):
    if r"https://" in url or r"http://" in url:
        return url
    else:
        return r"http://" + url

def heuristic_engine(content_html , content_text):

    returnable = []

    html_regex = r"(?i)<!DOCTYPE\s+html|<html\b|<head\b|<body\b|<title\b|<div\b|<span\b|<p\b"
    xml_regex = r'<\?xml\s+version="1\.\d+"\s*encoding="[^"]+"?\?>|<\w+(\s+\w+="[^"]*")*\s*\/?>|<\/\w+>'

    html_engine = re.compile(html_regex)
    xml_engine = re.compile(xml_regex)
    if html_engine.search(content_html):
        returnable.append("HTML")
    elif xml_engine.search(content_html):
        returnable.append("XML")
    else: 
        returnable.append("Not XML or HTML")

    encoding = UnicodeDammit(content_html).original_encoding
    returnable.append(encoding)

    lang_content = detect(content_text)
    returnable.append(lang_content)

    return returnable

def keyword_extraction(content):
    rake_nltk_var = Rake()
    rake_nltk_var.extract_keywords_from_text(content)
    return " ".join(rake_nltk_var.get_ranked_phrases())

def calculate_vector_similarity(text1 , text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def html_to_txt(html_content):
    soup = BeautifulSoup(html_content, features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()    
    text = soup.get_text()
    return text

def visual_similarity(website_1 , website_2):

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')

    service = Service(executable_path=ChromeDriverManager().install())

    driver = webdriver.Chrome(service=service, options=options)

    driver.get(website_1)
    time.sleep(5) #give it time to load
    driver.save_screenshot('website_1.png')
    driver.get(website_2)
    time.sleep(5) #give it time to load
    driver.save_screenshot('website_2.png')
    driver.quit()

    def convolution(image):

        kernel = np.array(( [0, 1, 0],
	                        [1, -4, 1],
	                        [0, 1, 0]), dtype="int")
        #laplacian kernel for getting the edges

        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]
        # huge thanks to guys over at https://pyimagesearch.com for this
        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")

        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                k = (roi * kernel).sum()
                output[y - pad, x - pad] = k

        output = rescale_intensity(output, in_range=(0, 255))
        output = (output * 255).astype("uint8")
        return output

    image_1 = cv2.imread("website_1.png")
    image_2 = cv2.imread("website_2.png")

    gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    convoled_1 = convolution(gray_1)
    convoled_2 = convolution(gray_2)

    convoled_2 = cv2.resize(convoled_2, (convoled_1.shape[1], convoled_1.shape[0]))
    similarity_index, diff = ssim(convoled_1, convoled_2, full=True)
    return round((similarity_index*100) , 3)


website_1 = url_safety(parse.website_1)
website_2 = url_safety(parse.website_2)

print(f"Website 1: {website_1} |  Website 2: {website_2}\n")

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0',
    "Accept-Encoding": "gzip, deflate, br, zstd"
    }

dom_content_1 = requests.get(website_1 , headers=headers).content.decode()
dom_content_2 = requests.get(website_2 , headers=headers).content.decode()

txt_content_1 = html_to_txt(dom_content_1)
txt_content_2 = html_to_txt(dom_content_2)

heur_1 = heuristic_engine(dom_content_1 , txt_content_1)
heur_2 = heuristic_engine(dom_content_1 , txt_content_1)

keywords_1 = keyword_extraction(txt_content_1)
keywords_2 = keyword_extraction(txt_content_2)



print(f"[HEURISTICS] Webpage 1 | Content type: {heur_1[0]} , Encoding: {heur_1[1]} , Language: {heur_1[2]}")
print(f"[HEURISTICS] Webpage 2 | Content type: {heur_2[0]} , Encoding: {heur_2[1]} , Language: {heur_2[2]}\n")
print(f"[LINGUISTICS] Vector Similarity between the websites code: {round(((calculate_vector_similarity(dom_content_1 , dom_content_2))*100),3)}%")
print(f"[LINGUISTICS] Vector Similarity between the websites content: {round(((calculate_vector_similarity(txt_content_1 , txt_content_2))*100),3)}%")
print(f"[LINGUISTICS] Vector Similarity between the websites keywords: {round(((calculate_vector_similarity(keywords_1 , keywords_2))*100),3)}%\n")

print(f"[VISUAL] Visual Similarity between the websites: {visual_similarity(website_1 , website_2)}%")
