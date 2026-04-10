from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json

nlp = spacy.load("en_core_web_sm")

SERP_API_KEY = os.environ.get("SERP_API_KEY")

def get_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:5000]
    except:
        return ""

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def handler(request):
    params = {
        "q": "4G吃到飽",
        "hl": "zh-tw",
        "gl": "tw",
        "google_domain": "google.com.tw",
        "api_key": SERP_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    organic = results.get("organic_results", [])[:10]

    urls = [r["link"] for r in organic if "link" in r]
    titles = [r["title"] for r in organic if "title" in r]

    texts = [get_text(u) for u in urls]
    entities_list = [extract_entities(t) for t in texts]

    # entity count
    entity_counts = [len(e) for e in entities_list]

    # clustering
    texts_clean = [" ".join([e[0] for e in ents]) for ents in entities_list]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts_clean)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    result = []
    for i in range(len(urls)):
        result.append({
            "title": titles[i],
            "url": urls[i],
            "entity_count": entity_counts[i],
            "cluster": int(clusters[i])
        })

    return {
        "statusCode": 200,
        "body": json.dumps(result, ensure_ascii=False)
    }
