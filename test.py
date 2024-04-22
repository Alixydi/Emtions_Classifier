from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_positive_sentiment():
    response = client.post("/predict", json={"text": "I love this product! It's amazing."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "joy"}  # Positive sentiment corresponds to joy

def test_negative_sentiment():
    response = client.post("/predict", json={"text": "This product is terrible. I regret buying it."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "anger"}  # Negative sentiment corresponds to anger

def test_neutral_sentiment():
    response = client.post("/predict", json={"text": "This product is okay. It meets my expectations."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "neutral"}  # Neutral sentiment remains unchanged

def test_anger_sentiment():
    response = client.post("/predict", json={"text": "This product makes me furious!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "anger"}  # Emotion of anger

def test_disgust_sentiment():
    response = client.post("/predict", json={"text": "I find this product revolting."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "disgust"}  # Emotion of disgust

def test_fear_sentiment():
    response = client.post("/predict", json={"text": "This product scares me."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "fear"}  # Emotion of fear

def test_sadness_sentiment():
    response = client.post("/predict", json={"text": "This product makes me really sad."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "sadness"}  # Emotion of sadness

def test_surprise_sentiment():
    response = client.post("/predict", json={"text": "I'm amazed by this product!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "surprise"}  # Emotion of surprise
