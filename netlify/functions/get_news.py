import json
import requests
from transformers import pipeline

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

def handler(event, context):
    try:
        # Parse query parameters
        query_params = event.get('queryStringParameters', {}) or {}
        category = query_params.get('category', 'general')
        page = int(query_params.get('page', 1))
        
        # Make request to News API
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
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'articles': [],
                    'has_more': False
                })
            }
        
        processed_articles = process_articles(articles_data)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'articles': processed_articles,
                'has_more': len(articles_data) == 10
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        } 