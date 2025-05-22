from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import requests
import os

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

app = Flask(__name__)

# Initialize the models
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def process_articles(articles_data):
    processed_articles = []
    for article in articles_data[:10]:
        text = article['title']
        classification = classifier(text)[0]
        summary = summarizer(text, max_length=40, min_length=20, do_sample=False)[0]['summary_text']
        
        processed_articles.append({
            'title': text,
            'confidence': f"{classification['score']*100:.1f}%",
            'is_fake': classification['label'] == 'LABEL_1',
            'summary': summary,
            'url': article.get('url', '#'),
            'image': article.get('urlToImage', '#'),
            'publishedAt': article.get('publishedAt', ''),
            'source': article.get('source', {}).get('name', 'Unknown')
        })
    return processed_articles

@app.route('/load_more', methods=['POST'])
def load_more():
    category = request.form.get('category')
    page = int(request.form.get('page', 1))
    
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "country": "us",
        "category": category,
        "apiKey": "4ae77d8cef5444f0a004714cd93a39f0",
        "pageSize": 10,
        "page": page
    }
    
    response = requests.get(url, params=params)
    articles_data = response.json()['articles']
    
    if not articles_data:
        return jsonify({'articles': [], 'has_more': False})
    
    processed_articles = process_articles(articles_data)
    return jsonify({
        'articles': processed_articles,
        'has_more': len(articles_data) == 10
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    articles = []
    current_category = None
    page = 1
    
    if request.method == 'POST':
        current_category = request.form.get('category', 'general')
        page = int(request.form.get('page', 1))
        
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            "country": "us",
            "category": current_category,
            "apiKey": "4ae77d8cef5444f0a004714cd93a39f0",
            "pageSize": 10,
            "page": page
        }
        
        response = requests.get(url, params=params)
        articles_data = response.json()['articles']
        articles = process_articles(articles_data)
    
    categories = [
        'business', 'entertainment', 'general', 'health',
        'science', 'sports', 'technology'
    ]
    
    return render_template('index.html', 
                         articles=articles, 
                         categories=categories,
                         current_category=current_category,
                         current_page=page)

if __name__ == '__main__':
    app.run(debug=True)
