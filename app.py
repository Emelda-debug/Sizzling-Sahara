import json
from flask import Flask, request, render_template, jsonify
from textblob import TextBlob
from dotenv import load_dotenv
from scipy import spatial 
import ast  
import openai 
from openai import OpenAI
import os
import pandas as pd
import re
import tiktoken
import plotly.io as pio
import plotly.express as px

REVIEWS_FILE = 'reviews.json'
RESERVATIONS_FILE = 'reservations.json'

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("OPEN_AI_KEY")
openai.api_key = api_key


embeddings_path = "emdeddings_dataset.csv"
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Search function
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    query_embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def detect_reservation_intent(user_input):
    reservation_keywords = ['reservation', 'book', 'reserve', 'booking']
    return any(keyword in user_input.lower() for keyword in reservation_keywords)

def handle_reservation_conversation():
    return "Sure! I can help you with that! Could you please provide your reservation details in the following format: Name, Date, Time, Number of People, Preferred Table and Table Number? e.g. (John Doe, 5 October, 7:00 PM, indoor table, 7 and 4 people)"

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below information from the Sizzlin Sahara Steakhouse Restaurant. Answer as a virtual assistant for the restaurant. Your name is Sizzla, you are a friendly female. Try your best to answer all the questions using the provided information. If the answer cannot be found in the info, write "I could not find a satisfactory answer for your question. Please, contact our Customer Service Assistant, Emelda, on +263 77 334 4079 or visit our website (https://sizzlinsahara.com) for more information."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nINFORMATION FOR Sizzlin Sahara Steakhouse Restaurant:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        message += next_article
    return message + question

def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about Sizzlin Sahara Steakhouse Restaurant."},
        {"role": "user", "content": message},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message

# Parse reservation details
def parse_reservation_details(text):
    pattern = r"(?P<name>[\w\s]+),\s*(?P<date>\d{1,2} \w+),\s*(?P<time>\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|\d{1,2}\s*(?:AM|PM|am|pm)?)\s*,\s*(?P<people>\d+)\s*people?\s*,\s*(?P<preferred_table>[\w\s]+),\s*table\s*(?P<table_number>\d+)"
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        return match.groupdict()
    else:
        return None





app = Flask(__name__)

# Route to display the interactive graph
@app.route('/show_reviews_graph')
def show_reviews_graph():
    reviews = load_reviews()
    
    sentiments = [analyze_sentiment(r['review']) for r in reviews]
    
    # Create a DataFrame for Plotly
    df = pd.DataFrame({'Sentiment': sentiments})
    
    # Create Plotly histogram
    fig = px.histogram(df, x='Sentiment', title='Review Sentiments', labels={'Sentiment': 'Sentiment'})
    
    # Convert Plotly graph to HTML
    graph_html = pio.to_html(fig, full_html=False)
    
    return render_template('reviews.html', graph_html=graph_html)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for reservation page
@app.route('/templates/reservation.html')
def reservation():
    return render_template('reservation.html')

# Route to handle chatbot questions
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Question parameter is required'}), 400
    
    # Detect the word 'reservation' in the question
    if detect_reservation_intent(question):
        return jsonify({'answer': handle_reservation_conversation()}), 200


    # Proceed with the usual chatbot logic for other queries
    response = ask(query=question)
    return jsonify({'answer': response}), 200



# Load and save reviews
def load_reviews():
    if os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE, 'r') as file:
            return json.load(file)
    return []

def save_reviews(reviews):
    with open(REVIEWS_FILE, 'w') as file:
        json.dump(reviews, file, indent=4)

# Function for sentiment analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    elif sentiment == 0:
        return 'neutral'
    else:
        return 'negative'

# Handle submitting reviews
@app.route('/submit_review', methods=['POST'])
def submit_review():
    data = request.get_json()
    review = data.get('review')

    if not review:
        return jsonify({'error': 'Review text is required'}), 400

    reviews = load_reviews()
    reviews.append({'review': review})

    # Analyze sentiment
    sentiment = analyze_sentiment(review)
    categorized_reviews = {
        'positive': [],
        'neutral': [],
        'negative': []
    }
    for rev in reviews:
        rev_sentiment = analyze_sentiment(rev['review'])
        categorized_reviews[rev_sentiment].append(rev)
    
    # Save categorized reviews
    for category, revs in categorized_reviews.items():
        with open(f'{category}_reviews.json', 'w') as file:
            json.dump(revs, file, indent=4)

    save_reviews(reviews)
    
    return jsonify({'message': 'Review submitted and analyzed successfully'}), 200



# Load and save reservations
def load_reservations():
    if os.path.exists(RESERVATIONS_FILE):
        with open(RESERVATIONS_FILE, 'r') as file:
            return json.load(file)
    return []

def save_reservations(reservations):
    with open(RESERVATIONS_FILE, 'w') as file:
        json.dump(reservations, file, indent=4)

# Route to make reservations
@app.route('/make_reservation', methods=['POST'])
def make_reservation(details=None):
    # If details are passed from the /ask route, use them
    if details:
        name = details.get('name')
        date = details.get('date')
        time = details.get('time')
        people = details.get('people')
        preferred_table = details.get('preferred_table')
        table_number = details.get('table_number')
    else:
        # Otherwise, check if it's a direct POST request with JSON data
        data = request.get_json()
        name = data.get('name')
        date = data.get('date')
        time = data.get('time')
        people = data.get('people')
        preferred_table = data.get('preferred_table')
        table_number = data.get('table_number')

    # Check if all required details are present
    if not name or not date or not time or not people or not preferred_table or not table_number:
        return jsonify({'error': 'All reservation details, including preferred table and table number, are required'}), 400

    # Create reservation object
    reservation = {
        'name': name,
        'date': date,
        'time': time,
        'people': people,
        'preferred_table': preferred_table,
        'table_number': table_number
    }

    # Load current reservations, append the new one, and save them
    reservations = load_reservations()
    reservations.append(reservation)
    save_reservations(reservations)

    # Generate confirmation message
    confirmation_message = confirm_reservation(reservation)

    return jsonify({'message': confirmation_message}), 200



def confirm_reservation(reservation):
    return (f"Thank you, {reservation['name']}! Your reservation for {reservation['people']} people on {reservation['date']} "
            f"at {reservation['time']} has been confirmed. Preferred table: {reservation['preferred_table']}, Table number: {reservation['table_number']}.")


if __name__ == '__main__':
    app.run(debug=True, port=8000)