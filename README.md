# Real-time Fake News Detection

A web application that detects fake news in real-time using machine learning. The application fetches news articles from various categories and analyzes them using a BERT-based model to determine their authenticity.

## Demo video
https://drive.google.com/file/d/174RspJM3vXSQ8uwyWhleb0DQUX9oiQL9/view?usp=sharing

## Features

- **Real-time News Analysis**: Fetches and analyzes news articles in real-time
- **Multiple Categories**: Supports various news categories:
  - Business
  - Entertainment
  - General
  - Health
  - Science
  - Sports
  - Technology
- **Dynamic Loading**: Load articles without page refresh
- **Confidence Scores**: Displays confidence measures for each classification
- **Article Summaries**: Provides concise summaries of articles
- **Source Information**: Shows article source and publication date
- **Visual Indicators**: Color-coded results for easy identification
  - Green: Real News
  - Red: Fake News

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd realtimefakenewsdetection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Select a news category from the dropdown menu

4. View the results:
   - Articles will be displayed with their classification
   - Each article shows:
     - Title
     - Source
     - Publication date
     - Summary
     - Confidence score
     - Classification (Real/Fake)

5. Load more articles:
   - Click the "Load More Articles" button at the bottom
   - New articles will be loaded dynamically

## Technical Details

- **Frontend**: HTML, CSS, JavaScript with Bootstrap 5
- **Backend**: Flask (Python)
- **ML Models**:
  - Fake News Detection: BERT-tiny model
  - Text Summarization: BART-large-CNN
- **News API**: NewsAPI.org

## Dependencies

- Flask: Web framework
- Transformers: Machine learning models
- Requests: HTTP client
- Torch: Deep learning framework
- Other supporting libraries (see requirements.txt)

## Note

The application uses the NewsAPI.org service. Make sure you have a valid API key configured in the application.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
