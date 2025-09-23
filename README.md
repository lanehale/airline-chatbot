# airline-chatbot
BERT is a large language model (LLM) and a foundation model designed for Natural Language Processing (NLP).
Training a pre-trained model is called fine-tuning.

For this project, I fine-tuned BERT to classify query types as status, booking or general queries with 98% accuracy.
To train BERT I created a number of queries and filled in random city, airport, and date variations using cities and airports from Kaggle dataset Airline_Delay_Cause.
I also created airline and airport mapping files from Airline_Delay_Cause for the chatbot to map user input to airlines and airports.

The chatbot can be used for real flight status and flight bookings at:
https://huggingface.co/spaces/lanehale1/Airline_Chatbot
