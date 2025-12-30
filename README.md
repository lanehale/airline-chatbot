# Airline Chatbot

An intelligent chatbot that uses natural language processing to handle flight inquiries, provide **real-time flight status**, and offer **current flight options with direct links to book on Expedia**.

👉 **Try the live demo**: [Hugging Face Spaces](https://huggingface.co/spaces/lanehale1/Airline_Chatbot)

![Chatbot Demo - Flight Search Results](https://raw.githubusercontent.com/lanehale/airline-chatbot/main/screenshots/flight-search-example.png)
![Chatbot Demo - Flight Status Response](https://raw.githubusercontent.com/lanehale/airline-chatbot/main/screenshots/flight-status-example.png)

*(Screenshots showing real-time flight options with Expedia booking links and live status updates)*

## Overview

This project is a practical airline customer service chatbot powered by NLP. It understands conversational queries, classifies user intent (flight status, booking/search, or general), and delivers accurate responses using live flight data.

Key features:
- Real-time flight status lookups
- Flight searches with up-to-date options and **direct Expedia booking links** (users complete actual bookings on Expedia)
- Supports multi-leg (round-trip) itineraries
- Robust natural language understanding for variations in cities, airports, dates, airlines, cabins, and passenger details

## Technical Details

- **Model**: Fine-tuned BERT (Bidirectional Encoder Representations from Transformers) for intent classification (status, booking/search, or general) — achieving **98% accuracy** on the test set.
- **Training Data**: Synthetically generated queries with randomized cities, airports, airlines, dates, and passenger configurations for realistic variation.
  - Base data sourced from the Kaggle [Airline Delay Cause](https://www.kaggle.com/datasets/giovamata/airlinedelaycauses) dataset.
  - Additional smaller international destinations and airlines added manually.
- **Entity Mapping**: Custom airline and airport mapping files created from the dataset (plus manual additions) to reliably recognize and normalize user input.

## Limitations & Future Work

- **Coverage**: Includes all major countries and many international destinations/airlines. Some less popular or niche destinations and smaller airlines are not yet supported.
- **Bookings**: Facilitated via direct links to Expedia (no in-chat payment or post-booking management).
- **Flight Status**: Powered by AeroAPI — status is available only for flights within approximately ±2 days of the current UTC date (based on the user's browser time). Requests outside this window cannot be fulfilled.
- **Planned improvements**:
  - Add support for multiple passengers and passenger types (e.g., adults, children, infants) as allowed by Expedia
  - Expand coverage to more niche destinations and airlines
  - Integrate a generative LLM for more natural, conversational responses
  - Handle additional intents (e.g., baggage policies, airport information)

## Acknowledgments

- BERT model and Transformers library from Hugging Face
- Flight data derived from public Kaggle datasets
- Real-time search and booking links powered by Expedia
- Flight status powered by AeroAPI (FlightAware)
