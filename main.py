from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nlp = spacy.load("en_core_web_sm")


def search_google(query):
    params = {
        "q": query,
        "hl": "zh-tw",
        "gl": "tw",
        "google_domain": "google.com.tw",
        "api_key": "你的SerpAPI KEY"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    organic = results.get("organic_results", [])[:5]

    return [
        {
            "title": r["title"],
            "url": r["link"]
        }
        for r in organic if "link" in r
    ]


def get_text(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        return " ".join([p.get_text() for p in soup.find_all("p")])[:3000]
    except:
        return ""


def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def analyze(query):
    results = search_google(query)

    texts = []
    for r in results:
        texts.append(get_text(r["url"]))

    entities = [extract_entities(t) for t in texts]

    cleaned = [" ".join([e[0] for e in ents]) for ents in entities]

    if len(cleaned) == 0:
        return {"error": "no data"}

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    output = []
    for i, r in enumerate(results):
        output.append({
            "title": r["title"],
            "url": r["url"],
            "cluster": int(clusters[i]) if i < len(clusters) else -1,
            "entity_count": len(entities[i]) if i < len(entities) else 0
        })

    return output
