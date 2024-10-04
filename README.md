# DOMS 
###### Document Object Model Similarity Analysis

### Analysis Steps
#### 1: Heuristics
Charset, language and code recognition
#### 2: Linguistic Analysis
Cosine similarity of the TFIDF text vectors of the code of the two websites and keyword analysis.
#### 3: Visual Analysis
Visual similarity of the website. Convolution of the images before similarity analysis to ignore minute changes

### How to use
```
python doms.py [-h] -w1 WEBSITE_1 -w2 WEBSITE_2
```
### Output example
![image](https://github.com/user-attachments/assets/be10c818-86c5-4b53-8ab1-0cf48bb60317)

### TO-DO
- Support for auth and proxies
- DOM hashing in chunks (Domhash but something better on my mind)
- Auto classification of some websites to known C2 panels
- Implement some similarity comparison myself instead of using scikit
- Something like YARA rules but for websites and better (maybe) + simpler

###### Made by Apollyon
