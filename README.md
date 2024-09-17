This README provides an overview of the Sizzlin Sahara Steakhouse Chatbot project, including its functionalities, dependencies, usage instructions, and contribution guidelines.

https://sizzling-sahara.onrender.com/
That is the link to the live site

Background
THis is a Sizzlin Sahara, a fictional restaurant that is afrocentric
The purpose of this website is to enhance customer experience on the site


Project Overview

This Flask-based chatbot utilizes OpenAI's API to answer customer questions about the Sizzlin Sahara Steakhouse Restaurant. It retrieves relevant information from a provided dataset and leverages TextBlob for sentiment analysis of user reviews. Additionally, it allows users to submit new reviews and displays a sentiment distribution graph.

Features

Answers questions about the Sizzlin Sahara Steakhouse Restaurant using OpenAI's text-embedding and GPT-3 models.
Analyzes sentiment of user reviews with TextBlob.
Allows users to submit new reviews.
Displays a graph of review sentiment distribution using Plotly.
Handles reservation inquiries and provides instructions for submitting reservations.


Dependencies

Flask
openai
dotenv
scipy
pandas
textblob
tiktoken
plotly.express
plotly.io
Installation

Clone this repository.
Create a virtual environment (recommended) and activate it.
Install the required dependencies using pip install -r requirements.txt.
Obtain an OpenAI API key and create a .env file with the following line: OPEN_AI_KEY=<your_api_key>.
Usage

Modify the following files according to your needs:
embeddings_dataset.csv: This CSV file should contain the restaurant information used by the chatbot.
reviews.json: This JSON file stores existing user reviews. (Optional)
Run the application using python app.py.
Access the chatbot interface at http://localhost:5000/.
Routes

/: Displays the chatbot interface.
/ask: Handles user questions submitted via a POST request.
/submit_review: Accepts user reviews for submission.
/show_reviews_graph: Displays the sentiment distribution graph of reviews.
/templates/reservation.html: Provides a page for reservation inquiries. (Functionality not implemented)
Contributing

We welcome contributions to this project! Please follow these guidelines:

Fork the repository.
Create a new branch for your changes.   
Implement your modifications and add tests if applicable.
Submit a pull request with a clear description of your changes.

Additional Notes

This is a basic implementation and can be further extended.
Consider error handling and security best practices for production deployments.
Explore advanced text-embedding and language models for enhanced functionality.
Contact

For any questions or feedback, feel free to create an issue on this repository.
