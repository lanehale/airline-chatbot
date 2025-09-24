# Copyright © 2025 Lane Hale. All rights reserved.

### Imports ###
import copy
import gradio as gr
import isodate
import json
import logging
import os
import pytz
import random
import re
import requests
import uuid

from datetime import datetime, date, timedelta
from dateutil import parser
from timezonefinder import TimezoneFinder
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Initialize TimezoneFinder once
timezone_finder = TimezoneFinder()

# Configure the root logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


### Prepare Airline, Airport Data ###

# Copy JSON Files to Dictionaries
file_path = "airline_code_to_name.json"

with open(file_path, "r") as file:
    carrier_codes = json.load(file)

file_path = "airline_name_and_code_to_code.json"

with open(file_path, "r") as file:
    carrier_dual_mapping = json.load(file)

file_path = "airport_code_to_name.json"

with open(file_path, "r") as file:
    airport_codes = json.load(file)

file_path = "airport_name_and_code_to_code.json"

with open(file_path, "r") as file:
    airport_dual_mapping = json.load(file)

# Sort carrier names/codes by length in descending order to prioritize longer matches
sorted_carriers = sorted(
    carrier_dual_mapping.items(), key=lambda item: len(item[0]), reverse=True
)

# Sort airport names/codes by length in descending order to prioritize longer matches
sorted_airport_mappings = sorted(
    airport_dual_mapping.items(), key=lambda item: len(item[0]), reverse=True
)


### Prepare Model Classification Pipeline ###

# Define ID to label mapping
id2label_mapping = {0: "booking", 1: "general", 2: "status"}

model_path = "best_checkpoint"

model = AutoModelForSequenceClassification.from_pretrained(
    model_path, id2label=id2label_mapping
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

question_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


################################################################################
###                               Functions                                  ###
################################################################################
#   choose_retry_phrase()                                                      #
#   choose_redundant_phrase()                                                  #
#   extract_date_info(query)                                                   #
#   extract_flight_info(query, intent)                                         #
#   parse_query_date(date_string)                                              #
#   get_flight_status(flight_ident)                                            #
#   get_flights_to_book(flight_info, passengers)                               #
#   get_timezone_name(latitude, longitude, tz_finder=timezone_finder)          #
#   get_local_datetime(datetime_string, local_timezone_str)                    #
#   append_departure_details(flight, response_lines)                           #
#   append_arrival_details(flight, response_lines)                             #
#   format_delay_time(delay_seconds)                                           #
#   build_status_response(flight_data, flight_ident)                           #
#   build_booking_response(booking_data, departure_date, tz_finder)            #
#   parse_multi_airport_response(user_input, flight_info_alt,                  #
#                                multi_airport_display_string)                 #
#   sort_flight_options(flight_options, sort_button)                           #
#   build_flight_options_batch(sorted_options, start_index, batch_size=10)     #
#   show_more_flight_options(chat_history, state_dict, sort_button)            #
#   chat_flight_assistant(user_input, chat_history, state_dict, sort_button)   #
#   clear_chat(chat_history, state_dict)                                       #
################################################################################

#############################################
#   Functions to Choose Phrase variations   #
#############################################
def choose_retry_phrase():
    retry_phrases = [
        "Please try again.",
        "Could you please retry?",
        "Could you try again?",
        "Please give it another try.",
        "Please try your request again.",
        "Could you please try again?",
    ]
    return random.choice(retry_phrases)

def choose_redundant_phrase():
    redundant_phrases = [
        "is redundant",
        "is superfluous",
        "is unnecessary",
        "isn't needed",
        "can be removed",
    ]
    return random.choice(redundant_phrases)


####################################
#   Extract Date Info from query   #
####################################
def extract_date_info(query):
    """
    Extracts dates in various formats from a natural language query.

    Args:
        query (str): The natural language query.

    Returns:
        dict: A dictionary containing the extracted date information.
    """
    date_dict = {}

    ### Breaking down this regex:

    # \b...\b       Ensures we match whole date patterns.
    # (?:...|...)   This is a non-capturing group that allows us to match one of several date formats.

    # \d{1,2}[-/]\d{1,2}[-/]\d{2,4}
    # Matches numeric date formats like MM/DD/YY, MM-DD-YYYY, M/D/YY, etc. with required year.
    #   \d{1,2}     One or two digits for month or day.
    #   [-/]        Matches either a hyphen or a slash as a separator.
    #   \d{2,4}     Two or four digits for the year.

    # \d{1,2}[-/]\d{1,2}
    # Matches numeric date formats like MM-DD or MM/DD.
    #   \d{1,2}     One or two digits for month or day.
    #   [-/]        Matches either a hyphen or a slash as a separator.

    # \b(?:Jan|Feb|...|December)\s+\d{1,2}(?:,\s*\d{4})?
    # Matches date formats with month names followed by day and optional year (e.g., Jan 8, Jan 8, 2023).
    #   \b(?:Jan|Feb|...)   Matches a month name (abbreviated or full) as a whole word.
    #   \s+                 One or more spaces after the month name.
    #   \d{1,2}             One or two digits for the day.
    #   (?:,*\s+\d{4})?     This part is optional (?) and matches optional comma, spaces, and a four-digit year.

    # \b\d{1,2}\s*(?:Jan|Feb|...|December)(?:,*\s+\d{4})?
    # Matches date formats with day followed by month name and optional year (e.g., 8 Jan, 8 Jan 2023, 8Jan 2023).
    #   \b\d{1,2}           One or two digits for the day.
    #   \s*                 Zero or more spaces.
    #   (?:Jan|Feb|...)     Matches a month name (abbreviated or full).
    #   (?:,*\s+\d{4})?     This part is optional (?) and matches optional comma, spaces, and a four-digit year.

    ###
    # Define the regex pattern as a single-line string for clarity
    ###
    date_regex = r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}[-/]\d{1,2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,*\s+\d{4})?|\b\d{1,2}\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)(?:,*\s+\d{4})?)\b"

    match = re.search(date_regex, query, re.IGNORECASE)

    if match:
        date_dict["date"] = match.group(0)
        # Replace the entire matched substring (group 0) with an empty string
        date_dict["query_without_date"] = re.sub(re.escape(match.group(0)), "", query)

    return date_dict


######################################
#   Extract Flight Info from query   #
######################################
def extract_flight_info(query, intent):
    """
    Extracts flight information from a natural language query.

    Args:
        query (str): The natural language query.
        intent (str): The intent of the query ('status' or 'booking').

    Returns:
        dict: A dictionary containing the extracted flight information.
        str: A string containing assistant responses.
    """
    assistant_response = ""
    errors = []

    #-----------------------------------------------------#
    #   Normalize common variations in city/place names   #
    #-----------------------------------------------------#
    # Create a dictionary of common variations to replace
    replacements = {
        r"\bSaint\b": "St.",
        r"\bSt\s": "St. ",
        r"\bFt\.\s*": "Fort ",
        r"\bFt\s": "Fort ",
        # Add other common variations as needed (e.g., 'Mount' to 'Mt.')
    }

    # Sort replacement patterns by length in descending order
    sorted_replacements = sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    )

    # Apply replacements to the query
    normalized_query = query
    for pattern, replacement in sorted_replacements:
        normalized_query = re.sub(
            pattern, replacement, normalized_query, flags=re.IGNORECASE
        )

    # Use the normalized_query for further processing
    query = normalized_query

    #---------------------------------------------------------------------------#
    #   Status: Extract airline code (e.g., AA) and flight number (e.g., 123)   #
    #---------------------------------------------------------------------------#
    if intent == "status":

        flight_info = {"airline_code": None, "flight_number": None, "flight_date": None}

        # Look for date info and remove it from the query
        date_info = extract_date_info(query)

        if "date" in date_info:
            query = date_info["query_without_date"]
            flight_info["flight_date"] = date_info["date"]

        # Remove hyphens and slashes from the query
        query_split = re.split(r"[-/]", query)
        query = " ".join(query_split)

        for name_or_code, codes in sorted_carriers:
            # Use regex to find whole word matches for names and codes
            pattern = r"\b" + re.escape(name_or_code) + r"\b"
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                flight_info["airline_code"] = codes
                break  # Once an airline is found, we can stop looking for others

        if flight_info["airline_code"] is None:
            # Fallback to matching AA123 format if no specific name/code match found
            match = re.search(r"\b([A-Z]{2})(\d{1,4})\b", query, re.IGNORECASE)
            if match:
                flight_info["airline_code"] = match.group(
                    1
                ).upper()  # Ensure airline code is uppercase
                flight_info["flight_number"] = match.group(2)

        if flight_info["flight_number"] is None:
            # If an airline code was found by name/code, now look for the flight number
            match = re.search(r"\d{1,4}", query)
            if match:
                flight_info["flight_number"] = match.group(0)

    #---------------------------------------------------------------------------------#
    #   Booking: Extract city or airport codes (e.g., Denver to Chicago → DEN, ORD)   #
    #---------------------------------------------------------------------------------#
    elif intent == "booking":

        # Parse date info and remove it from the query
        date_info = extract_date_info(query)
        if "date" in date_info:
            query = date_info["query_without_date"]
        else:
            # No date was found
            errors.append("- I didn't find a valid departure date in your entry.")
            errors.append(
                "- Try a format like 6-23, 6/23, or Jun 23 (year is optional for any format)."
            )

        # Remove hyphens and slashes from the query
        query_split = re.split(r"[-/]", query)
        query = " ".join(query_split)

        found_locations = []
        temp_query = query

        for name_or_code, codes in sorted_airport_mappings:
            # Use regex to find whole word matches for names and codes
            pattern = r"\b" + re.escape(name_or_code) + r"\b"
            match = re.search(pattern, temp_query, re.IGNORECASE)

            if match:
                # Determine if the match is likely an airport code (3 capital letters)
                is_code = bool(re.fullmatch(r"[A-Z]{3}", name_or_code))

                # Add all codes associated with the matched name/code
                for code in codes:
                    found_locations.append(
                        {
                            "code": code,
                            "name": name_or_code,
                            "index": match.start(),
                            "is_code": is_code,
                        }
                    )
                # Replace the matched substring with unmatchable text
                start_index = match.start()
                end_index = match.end()
                replacement = "__" + codes[0] + "__"
                temp_query = (
                    temp_query[:start_index] + replacement + temp_query[end_index:]
                )

        # Sort found locations by their index in the query to determine origin and destination
        found_locations.sort(key=lambda x: x["index"])

        origin = {}
        destination = {}

        # Assign locations to origin and destination based on order in query
        for location in found_locations:
            airport_code = location["code"]
            airport_info = airport_codes[airport_code]

            # Fetch full city, state, airport names from airport_codes
            airport_name = ""
            if airport_info:
                # Split by comma and colon for "City, State: Airport Name"
                airport_info_parts = [
                    part.strip() for part in re.split(r"[,:]", airport_info)
                ]
                airport_state = airport_info_parts[1]
                airport_name = airport_info_parts[2]

            airport_dict = {
                "code": airport_code,
                "name": airport_name,
                "state": airport_state,
            }
            city_name = location["name"]

            if origin == {}:
                origin["city"] = city_name
                origin["airport"] = [airport_dict]
                origin["is_code"] = location["is_code"]
            elif city_name == origin["city"]:
                # If it's the same city as origin, add this airport to the origin list
                origin["airport"].append(airport_dict)
            elif destination == {}:
                destination["city"] = city_name
                destination["airport"] = [airport_dict]
                destination["is_code"] = location["is_code"]
            elif city_name == destination["city"]:
                # If it's the same city as destination, add this airport to the destination list
                destination["airport"].append(airport_dict)

        if origin == {}:
            errors.append(
                "- I didn't find two valid city names or airport codes in your entry."
            )
        elif destination == {}:
            if found_locations[0]["is_code"]:
                errors.append(
                    f"- I found '{found_locations[0]['code']}' in your entry, but a second valid city name or airport code was missing."
                )
            else:
                errors.append(
                    f"- I found '{origin['city']}' in your entry, but a second valid city name or airport code was missing."
                )

        if errors:
            # Update assistant_response with the error messages
            errors.insert(0, "Sorry, I couldn't understand your booking request:")
            errors.append("")
            errors.append(choose_retry_phrase())
            assistant_response = "\n".join(errors)
            flight_info = {}

        else:
            # Expedia cabin options can be economy/first/business/premiumeconomy
            cabin = "economy"  # Default cabin
            query_lower = query.lower()

            premium_economy_phrase = [
                "premium economy",
                "premium econ",
                "prem econ",
                "prem economy",
            ]
            business_phrase = ["business", "bus.", "biz"]
            first_phrase = ["first class", "1st class", "in first", "in 1st"]

            if any(phrase in query_lower for phrase in premium_economy_phrase):
                cabin = "premiumeconomy"
            elif any(phrase in query_lower for phrase in business_phrase):
                cabin = "business"
            elif any(phrase in query_lower for phrase in first_phrase):
                cabin = "first"

            flight_info = {
                "origin": origin,
                "destination": destination,
                "date": date_info["date"],
                "cabin": cabin,
            }

    #-----------------------------#
    #   General: return nothing   #
    #-----------------------------#
    else:
        flight_info = {}

    return flight_info, assistant_response


########################
#   Parse Query Date   #
########################
def parse_query_date(date_string):
    """
    Parses a date string into a datetime object using dateutil.
    If the date string does not include a year, it defaults to the current year.
    If the resulting date is before today's date (excluding time),
    it increments the year to handle future dates entered without a year.

    Args:
        date_string (str): The date string to parse.

    Returns:
        datetime: A datetime object representing the parsed date, or None if parsing fails.
    """
    try:
        # Get today's date without time for comparison
        today_date = date.today()

        # Parse the date string. By default, dateutil will use the current year if none is provided.
        parsed_datetime = parser.parse(date_string)

        # If the parsed date is before today's date, increment the year.
        # We compare only the date part to ignore the time component.
        if parsed_datetime.date() < today_date:
            parsed_datetime = parsed_datetime.replace(year=parsed_datetime.year + 1)

        return parsed_datetime

    except (ValueError, TypeError) as e:
        # Handle cases where parsing might fail
        logging.error(f"parse_query_date: Could not parse date string: {date_string}. Error: {e}")
        return None


##################################
#   Get Flight Status from API   #
##################################
def get_flight_status(flight_ident):
    """
    Requests flight status for a specific flight using today's date in UTC.

    Args:
        flight_ident (str): The flight identifier (e.g., 'WN1905').

    Returns:
        dict: A dictionary containing flight status information.
    """
    api_key = os.environ.get("AEROAPI_KEY")
    base_url = "https://aeroapi.flightaware.com/aeroapi/"

    # Get today's date in UTC
    now_utc = datetime.now(pytz.utc)
    # Go back 2 days UTC for long international flights
    start_date = (now_utc - timedelta(days=2)).strftime("%Y-%m-%d")
    # With no end_date the API defaults to its max of 2 days after today's date

    # Define headers for authentication
    headers = {"x-apikey": api_key}

    # Define parameters for the API request
    params = {
        "ident_type": "designator",
        "start": start_date,
    }

    url = base_url + "flights/" + flight_ident

    # Make the request
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        flight_data = response.json()
    else:
        logging.error(f"Error fetching flight status: {response.status_code} - {response.text}")
        flight_data = None

    return flight_data


####################################
#   Get Flights to Book from API   #
####################################
def get_flights_to_book(flight_info, passengers, state_dict):
    """
    Requests flight listings for a requested origin to destination and date.
    Optional request parameters:
        cabin (str): The cabin type (e.g., 'economy')
        passengers (dict): A dictionary containing passenger information.

    Args:
        flight_info (dict): A dictionary containing flight information.
        passengers (dict): A dictionary containing passenger information.
        state_dict (dict): A dictionary containing state information.

    Returns:
        dict: A dictionary containing flight listings.
    """
    LIMIT = "50"
    expedia_key = os.environ.get("EXPEDIA_KEY")
    expedia_auth = os.environ.get("EXPEDIA_AUTH")
    my_company_name = "ActualIntelligence" + "-"
    chat_number = 1  # We may increment this later using sqlite
    chat_number_str = str(chat_number) + "-"
    transaction_id = str(uuid.uuid4())
    partner_transaction_id = my_company_name + chat_number_str + transaction_id
    request_url = "https://apim.expedia.com/flights/listings"

    # Define headers for authentication
    headers = {
        "Accept": "application/vnd.exp-flight.v3+json",
        "Partner-Transaction-Id": partner_transaction_id,
        "Key": expedia_key,
        "Authorization": expedia_auth,
    }

    # Define parameters for the API request
    params = {}

    """ add check for 6 passengers max later """
    if passengers["adult"] > 0:
        params["adult"] = str(passengers["adult"])
    if passengers["senior"] > 0:
        params["senior"] = str(passengers["senior"])
    if passengers["childrenAges"][0] > 0:
        params["childrenAges"] = str(passengers["childrenAges"])
    if passengers["infantInLap"] > 0:
        params["infantInLap"] = str(passengers["infantInLap"])
    if passengers["infantInSeat"] > 0:
        params["infantInSeat"] = str(passengers["infantInSeat"])

    params["cabinClass"] = flight_info["cabin"]

    # params["numberOfStops"] = 0

    params["limit"] = LIMIT

    # selectedCarriers = ['AA', 'DL', 'UA']

    # 'all_airports' is only set to True when the user enters 'all' on a multi-airport
    # prompt, and 'multi_airport_prompt_active' is then set to False.
    # Therefore, when 'multi_airport_prompt_active' is True, 'all_airports' is False.
    if state_dict["all_airports"]:
    	# Use city name, state if origin has multi-airports
        if len(flight_info["origin"]["airport"]) > 1:
            origin = (
                flight_info["origin"]["city"]
                + ", "
                + flight_info["origin"]["airport"][0]["state"]
            )
        else:
            origin = flight_info["origin"]["airport"][0]["code"]

    	# Use city name, state if destination has multi-airports
        if (
            state_dict["all_airports"]
            and len(flight_info["destination"]["airport"]) > 1
        ):
            destination = (
                flight_info["destination"]["city"]
                + ", "
                + flight_info["destination"]["airport"][0]["state"]
            )
        else:
            destination = flight_info["destination"]["airport"][0]["code"]
    else:
        # Use airport codes
        origin = flight_info["origin"]["airport"][0]["code"]
        destination = flight_info["destination"]["airport"][0]["code"]

    	# Set nearby airports filter if airport code(s) was input
    	# or selected after multi-airport prompt
        if (
            state_dict["multi_airport_prompt_active"]
            or flight_info["origin"]["is_code"]
            or flight_info["destination"]["is_code"]
        ):
            params["filterNearByAirport"] = True

    requested_date = parse_query_date(flight_info["date"])
    departure_date = requested_date.strftime("%Y-%m-%d")

    origin_destination_pairs = [
        {"origin": origin, "destination": destination, "departureDate": departure_date},
    ]

    for i, pair in enumerate(origin_destination_pairs):
        segment = "segment" + str(i + 1)
        params[segment + ".origin"] = pair["origin"]
        params[segment + ".destination"] = pair["destination"]
        params[segment + ".departureDate"] = pair["departureDate"]

    # Make the request
    response = requests.get(request_url, headers=headers, params=params)

    if response.status_code == 200:
        flight_data = response.json()
    else:
        logging.error(f"Error fetching flights to book: {response.status_code} - {response.text}")
        flight_data = None

    return flight_data


#########################
#   Get Timezone Name   #
#########################
def get_timezone_name(latitude, longitude, tz_finder=timezone_finder):
    """
    Finds the IANA timezone name for a given latitude and longitude using a
    pre-initialized TimezoneFinder instance.

    Args:
        latitude (str or float): The latitude of the location.
        longitude (str or float): The longitude of the location.
        tz_finder (TimezoneFinder): A pre-initialized TimezoneFinder object.
                                    Defaults to the globally initialized timezone_finder.

    Returns:
        str: The IANA timezone name (e.g., 'America/New_York'), or None if not found.
    """
    try:
        # timezonefinder expects floats
        latitude = float(latitude)
        longitude = float(longitude)
        timezone_name = tz_finder.timezone_at(lng=longitude, lat=latitude)
        return timezone_name
    except (ValueError, TypeError) as e:
        logging.error(f"Error finding timezone for lat={latitude}, lon={longitude}: {e}")
        return None


##########################
#   Get Local DateTime   #
##########################
def get_local_datetime(datetime_string, local_timezone_str):
    """
    Converts a datetime string (which can include UTC or an offset) to a timezone-aware local datetime object.

    Args:
        datetime_string (str): The datetime string to parse (e.g., 'YYYY-MM-DDTHH:MM:SSZ' or 'YYYY-MM-DDTHH:MM:SS±HH:MM').
        local_timezone_str (str): The target local timezone string (e.g., 'America/New_York').

    Returns:
        datetime: A timezone-aware local datetime object, or None if parsing or timezone conversion fails.
    """
    try:
        # Use dateutil.parser.parse to handle both 'Z' and offset formats
        gmt_datetime_utc = parser.parse(datetime_string)

        # Ensure the parsed datetime is timezone-aware (it should be with dateutil for these formats)
        if gmt_datetime_utc.tzinfo is None:
            # If for some reason it's not timezone-aware, assume UTC if it ends with Z
            if datetime_string.endswith("Z"):
                gmt_datetime_utc = gmt_datetime_utc.replace(tzinfo=pytz.utc)
            else:
                # If no timezone info and not ending in Z, we can't proceed reliably
                logging.warning(
                    f"Warning: Parsed datetime string has no timezone information: {datetime_string}"
                )
                return None

        # Get the local timezone
        try:
            local_timezone = pytz.timezone(local_timezone_str)
        except pytz.UnknownTimeZoneError:
            logging.error(f"Error: Unknown timezone: {local_timezone_str}")
            return None

        # Convert the UTC datetime to the local timezone
        local_datetime = gmt_datetime_utc.astimezone(local_timezone)

        return local_datetime

    except (ValueError, TypeError) as e:
        logging.error(f"get_local_datetime: Could not parse datetime string: {datetime_string}. Error: {e}")
        return None


################################
#   Append Departure Details   #
################################
def append_departure_details(flight, response_lines):
    """
    Appends flight departure details to the response lines.

    Args:
        flight (dict): A dictionary containing flight details.
        response_lines (list): A list to store response lines.
    """
    scheduled_departure_local_datetime = get_local_datetime(
        flight["scheduled_out"], flight["origin"]["timezone"]
    )
    scheduled_departure_local_datetime_str = (
        scheduled_departure_local_datetime.strftime("%H:%M %Z %b-%d")
    )

    arrival_time_str = flight["estimated_in"] or flight["scheduled_in"]
    scheduled_arrival_local_datetime = get_local_datetime(
        arrival_time_str, flight["destination"]["timezone"]
    )
    scheduled_arrival_local_datetime_str = scheduled_arrival_local_datetime.strftime(
        "%H:%M %Z %b-%d"
    )

    response_lines.append(
        f"  Origin: {flight['origin']['city']} ({flight['origin']['code_iata']})"
    )
    response_lines.append(
        f"  Destination: {flight['destination']['city']} ({flight['destination']['code_iata']})"
    )
    response_lines.append(
        f"  Scheduled departure: {scheduled_departure_local_datetime_str}"
    )
    response_lines.append(
        f"  Estimated arrival: {scheduled_arrival_local_datetime_str}"
    )

    departure_gate = flight["gate_origin"]
    departure_terminal = flight["terminal_origin"]
    if departure_gate is None and departure_terminal is None:
        response_lines.append(f"  Gate, Terminal (not avail)")
    else:
        if departure_gate is None:
            departure_gate = "(not avail)"
        if departure_terminal is None:
            departure_terminal = "(not avail)"
        response_lines.append(f"  Gate {departure_gate} Terminal {departure_terminal}")


##############################
#   Append Arrival Details   #
##############################
def append_arrival_details(flight, response_lines):
    """
    Appends flight arrival details to the response lines.

    Args:
        flight (dict): A dictionary containing flight details.
        response_lines (list): A list to store response lines.
    """
    scheduled_arrival_local_datetime = get_local_datetime(
        flight["scheduled_in"], flight["destination"]["timezone"]
    )
    scheduled_arrival_local_datetime_str = scheduled_arrival_local_datetime.strftime(
        "%H:%M %Z %b-%d"
    )

    response_lines.append(
        f"  Origin: {flight['origin']['city']} ({flight['origin']['code_iata']})"
    )
    response_lines.append(
        f"  Destination: {flight['destination']['city']} ({flight['destination']['code_iata']})"
    )
    response_lines.append(
        f"  Scheduled arrival: {scheduled_arrival_local_datetime_str}"
    )


#########################
#   Format Delay Time   #
#########################
def format_delay_time(delay_seconds):
    """
    Converts a delay time in seconds to a human-readable format.

    Args:
        delay_seconds (int): The delay time in seconds.

    Returns:
        str: A string representing the delay time in a human-readable format.
    """
    # Use the absolute value for divmod
    hours, remainder = divmod(abs(delay_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    hours_text = f"{hours} hour{'s' if hours != 1 else ''}"
    minutes_text = f"{minutes} minute{'s' if minutes != 1 else ''}"
    seconds_text = f"{seconds} second{'s' if seconds != 1 else ''}"

    if hours != 0:
        if minutes != 0:
            return hours_text + " " + minutes_text
        else:
            return hours_text
    elif minutes != 0:
        # Unexpected, delay seconds seem to always be multiples of 60
        if seconds != 0:
            return minutes_text + " " + seconds_text
        else:
            return minutes_text
    else:
        # Unexpected, delay seconds seem to always be multiples of 60
        return "less than a minute"


#############################
#   Build Status Response   #
#############################
def build_status_response(flight_data, flight_ident):
    """
    Builds a string containing flight status for a specific flight.
    Filters flights based on arrival date in the destination timezone.

    Args:
        flight_data (dict): A dictionary containing flight status information.
        flight_ident (str): The flight identifier (e.g., 'WN1905').

    Returns:
        str: A string containing the formatted flight status, or a message if no data.
    """
    response_lines = []  # List to build the response string
    relevant_flights = []  # List to store flights relevant to the arrival date

    if flight_data and flight_data.get("flights"):
        # Flight data is returned newest to oldest
        diverted_flight = False
        airline_code = flight_ident[:2]
        flight_number = flight_ident[2:]
        # Defaults to airline code if name isn't found
        airline_name = carrier_codes.get(airline_code, airline_code)
        flight_ident_text = airline_name + " flight " + flight_number

        now_utc = datetime.now(pytz.utc)  # Get today's datetime in UTC

        destination_timezone_str = flight_data["flights"][0]["destination"]["timezone"]
        destination_timezone = pytz.timezone(destination_timezone_str)
        now_local_dest_datetime = now_utc.astimezone(
            destination_timezone
        )  # Get today's local datetime of destination
        target_arrival_date = now_local_dest_datetime.date()
        target_arrival_date_plus_one = target_arrival_date + timedelta(days=1)

        origin_timezone_str = flight_data["flights"][0]["origin"]["timezone"]
        origin_timezone = pytz.timezone(origin_timezone_str)
        now_local_origin_datetime = now_utc.astimezone(
            origin_timezone
        )  # Get today's local datetime of origin

        # Filter flights based on arrival date in the destination timezone
        for flight in flight_data["flights"]:

            # Parse the relevant arrival time string (actual > estimated > scheduled)
            arrival_time_str = (
                flight["actual_in"] or flight["estimated_in"] or flight["scheduled_in"]
            )

            if arrival_time_str:
                arrival_datetime_local = get_local_datetime(
                    arrival_time_str, destination_timezone_str
                )
                arrival_date_local = arrival_datetime_local.date()

                # Check if the arrival date is today or the next day
                # in the destination timezone
                if (
                    arrival_date_local == target_arrival_date
                    or arrival_date_local == target_arrival_date_plus_one
                ):
                    flight["arrival_datetime_local"] = (
                        arrival_datetime_local  # Save for later
                    )
                    relevant_flights.append(flight)

        # Check relevant flights for status
        relevant_flights_length = len(relevant_flights)

        for i in range(relevant_flights_length):
            flight = relevant_flights[i]

            if relevant_flights_length > 1 and i + 1 < relevant_flights_length:
                # Each next flight in the list is an earlier datetime than the last
                next_flight = relevant_flights[i + 1]

                if next_flight["diverted"]:
                    diverted_flight = True
                    response_lines.append(f"{flight_ident_text} is diverted.\n")
                    append_arrival_details(next_flight, response_lines)

                    if flight["fa_flight_id"] != next_flight["fa_flight_id"]:
                        response_lines.append(
                            "\nNo diverted flight data available to display."
                        )
                        break

                else:
                    # Process the flight closest to the scheduled local arrival
                    # or departure time compared to now's time
                    arrival_time_difference = abs(
                        now_local_dest_datetime - next_flight["arrival_datetime_local"]
                    )

                    # Parse the relevant departure time string (actual > estimated > scheduled)
                    departure_time_str = (
                        flight["actual_out"]
                        or flight["estimated_out"]
                        or flight["scheduled_out"]
                    )
                    departure_datetime_local = get_local_datetime(
                        departure_time_str, origin_timezone_str
                    )
                    departure_time_difference = abs(
                        now_local_origin_datetime - departure_datetime_local
                    )

                    # Is the earlier flight's arrival closer to 'now' than
                    # the later flight's departure?
                    if arrival_time_difference < departure_time_difference:
                        flight = next_flight

            if flight["cancelled"]:
                response_lines.append(f"{flight_ident_text} is cancelled.\n")
                append_arrival_details(flight, response_lines)
                scheduled_departure_local_datetime = get_local_datetime(
                    flight["scheduled_out"], flight["origin"]["timezone"]
                )
                scheduled_departure_local_datetime_str = (
                    scheduled_departure_local_datetime.strftime("%H:%M %Z %b-%d")
                )
                response_lines.insert(
                    -1,
                    f"  Scheduled departure: {scheduled_departure_local_datetime_str}",
                )
                break

            # Flight hasn't departed yet
            if flight["actual_out"] is None:
                departure_time_str = flight["estimated_out"] or flight["scheduled_out"]
                departure_datetime_local = get_local_datetime(
                    departure_time_str, origin_timezone_str
                )
                local_departure_datetime_str = departure_datetime_local.strftime(
                    "%H:%M %Z %b-%d"
                )
                local_departure_time = departure_datetime_local.strftime("%H:%M %Z")
                departure_delay_seconds = flight["departure_delay"]

                if departure_delay_seconds == 0:
                    response_lines.append(
                        f"{flight_ident_text} estimated departure on time at {local_departure_time}.\n"
                    )
                else:
                    delay_time_text = format_delay_time(departure_delay_seconds)

                    if departure_delay_seconds > 0:
                        response_lines.append(
                            f"{flight_ident_text} estimated departure {delay_time_text} late at {local_departure_time}.\n"
                        )
                    else:
                        response_lines.append(
                            f"{flight_ident_text} estimated departure {delay_time_text} early at {local_departure_time}.\n"
                        )

                append_departure_details(flight, response_lines)
                break

            # Flight has departed
            if flight["actual_in"] is not None:
                actual_or_estimated_text = "arrived"
            else:
                actual_or_estimated_text = "estimated arrival"

            local_datetime = flight["arrival_datetime_local"]
            local_arrival_datetime_str = local_datetime.strftime("%H:%M %Z %b-%d")
            local_arrival_time = local_datetime.strftime("%H:%M %Z")
            arrival_delay_seconds = flight["arrival_delay"]

            if diverted_flight:
                response_lines.append("\nThe diverted flight data follows:\n")

            if arrival_delay_seconds == 0:
                response_lines.append(
                    f"{flight_ident_text} {actual_or_estimated_text} on time at {local_arrival_time}.\n"
                )
            else:
                delay_time_text = format_delay_time(arrival_delay_seconds)

                if arrival_delay_seconds > 0:
                    response_lines.append(
                        f"{flight_ident_text} {actual_or_estimated_text} {delay_time_text} late at {local_arrival_time}.\n"
                    )
                else:
                    response_lines.append(
                        f"{flight_ident_text} {actual_or_estimated_text} {delay_time_text} early at {local_arrival_time}.\n"
                    )

            append_arrival_details(flight, response_lines)

            if actual_or_estimated_text == "arrived":
                response_lines.append(
                    f"  Actual arrival:    {local_arrival_datetime_str}"
                )
            else:
                response_lines.append(
                    f"  Estimated arrival: {local_arrival_datetime_str}"
                )

            arrival_gate = flight["gate_destination"]
            arrival_terminal = flight["terminal_destination"]
            if arrival_gate is None and arrival_terminal is None:
                response_lines.append(f"  Gate, Terminal (not avail)")
            else:
                if arrival_gate is None:
                    arrival_gate = "(not avail)"
                if arrival_terminal is None:
                    arrival_terminal = "(not avail)"
                response_lines.append(
                    f"  Gate {arrival_gate} Terminal {arrival_terminal}"
                )

            baggage_claim = flight["baggage_claim"]
            if baggage_claim is None:
                response_lines.append(f"  Baggage Claim (not avail)")
            else:
                response_lines.append(f"  Baggage Claim {baggage_claim}")

            break

        # End of relevant_flights for loop

    if response_lines == []:
        response_lines.append("No flight data available to display.")

    return "\n".join(response_lines)  # Join the lines into a single string


##############################
#   Build Booking Response   #
##############################
def build_booking_response(booking_data, departure_date, tz_finder):
    """
    Builds a string containing booking details.

    Args:
        booking_data (dict): A dictionary containing booking data.
        departure_date (datetime): The departure date.
        tz_finder (TimezoneFinder): An instance of TimezoneFinder for timezone lookup.

    Returns:
        header_string (str): A string containing the formatted booking header.
        flight_options_list (list): A list of dictionaries containing flight options.
    """
    # Use a list of dictionaries to save response lines
    # and sorting keys for each Offer
    flight_options_list = []

    if booking_data and booking_data.get("Offers"):
        departure_date_str = departure_date.strftime("%b %d, %Y")
        leg_index = len(booking_data["Segments"][0]["Legs"]) - 1
        origin_city = booking_data["Segments"][0]["Legs"][0]["DepartureAirport"]["City"]
        dest_city = booking_data["Segments"][0]["Legs"][leg_index]["ArrivalAirport"][
            "City"
        ]
        header_string = f"{origin_city} to {dest_city} on {departure_date_str}"

        for i, offer in enumerate(booking_data["Offers"]):
            response_lines = []  # Use a list to build the response string
            segment_ids_list = offer["SegmentIds"]

            for segment_id in segment_ids_list:
                for segment in booking_data["Segments"]:
                    if segment["SegmentId"] == segment_id:
                        leg_index = len(segment["Legs"]) - 1

                        # Get leg data
                        for leg in segment["Legs"]:
                            origin_city = leg["DepartureAirport"]["City"]
                            origin_code = leg["DepartureAirport"]["Code"]
                            dest_city = leg["ArrivalAirport"]["City"]
                            dest_code = leg["ArrivalAirport"]["Code"]
                            origin = f"{origin_city} ({origin_code})"
                            destination = f"{dest_city} ({dest_code})"
                            response_lines.append(
                                f"  {leg['MarketingAirlineName']} ({leg['MarketingAirlineCode']} {leg['FlightNumber']} {origin} - {destination})"
                            )

                        # Get origin timezone once per offer/segment
                        origin_timezone = get_timezone_name(
                            segment["Legs"][0]["DepartureAirport"]["Latitude"],
                            segment["Legs"][0]["DepartureAirport"]["Longitude"],
                            tz_finder,
                        )
                        # Fallback to UTC if timezone not found
                        if origin_timezone is None:
                            origin_timezone = "UTC"
                        local_departure_datetime = get_local_datetime(
                            segment["Legs"][0]["DepartureDateTime"], origin_timezone
                        )
                        # Format the datetime object into a 12-hour string with AM/PM
                        local_departure_time = local_departure_datetime.strftime(
                            "%I:%M %p"
                        )
                        # Extract the time object to use in sorting later
                        local_dep_time_object = local_departure_datetime.time()

                        # Get destination timezone once per offer/segment
                        destination_timezone = get_timezone_name(
                            segment["Legs"][leg_index]["ArrivalAirport"]["Latitude"],
                            segment["Legs"][leg_index]["ArrivalAirport"]["Longitude"],
                            tz_finder,
                        )
                        # Fallback to UTC if timezone not found
                        if destination_timezone is None:
                            destination_timezone = "UTC"
                        local_arrival_datetime = get_local_datetime(
                            segment["Legs"][leg_index]["ArrivalDateTime"],
                            destination_timezone,
                        )
                        # Format the datetime object into a 12-hour string with AM/PM
                        local_arrival_time = local_arrival_datetime.strftime("%I:%M %p")

                        duration_timedelta = isodate.parse_duration(
                            segment["FlightDuration"]
                        )
                        total_seconds = duration_timedelta.total_seconds()
                        hours, remainder = divmod(total_seconds, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        duration_text = f"{int(hours)}h {int(minutes)}m"

                        total_stops = segment["TotalStops"]
                        if total_stops == 0:
                            total_stops_text = "nonstop"
                        else:
                            total_stops_text = (
                                f"{total_stops} stop{'s' if total_stops > 1 else ''}"
                            )

                        response_lines.append(
                            f"  {local_departure_time} - {local_arrival_time} ({duration_text}, {total_stops_text})"
                        )

            total_price = offer["OfferPrice"]["TotalPrice"][
                "Value"
            ]  # Access the Value key
            currency = offer["OfferPrice"]["TotalPrice"][
                "Currency"
            ]  # Access the Currency key
            url = offer["Links"]["WebDetails"]["Href"]
            link_text = "View Details and Book"

            if currency == "USD":
                # Use HTML anchor tag for link
                response_lines.append(
                    f"  ${total_price}  <a href='{url}'>{link_text}</a>"
                )
            else:
                response_lines.append(
                    f"  {total_price} {currency}  <a href='{url}'>{link_text}</a>"
                )

            inner_dict = {
                "price": total_price,
                "time": local_dep_time_object,
                "duration": total_seconds,
            }
            # Join the lines into a single string with HTML line breaks
            inner_dict["display_text"] = "<br>".join(response_lines)
            flight_options_list.append(inner_dict)

    else:
        header_string = "No flights available to display."

    return header_string, flight_options_list


####################################
#   Parse Multi-Airport Response   #
####################################
def parse_multi_airport_response(
    user_input, state_dict, flight_info_alt, multi_airport_display_string
):
    """
    Parses the airport code(s) or line number(s) in the follow-up user response,
    and updates flight_info_alt with the selected airport code(s).

    Args:
        user_input (str): The user's message.
        state_dict (dict): The state dictionary.
        flight_info_alt (dict): The flight information dictionary.
        multi_airport_display_string (str): The multi-airport display string.

    Returns:
        flight_info_alt (dict): The updated flight information dictionary.
        assistant_response (str): The assistant's response.
    """
    assistant_response = ""

    if "all" in user_input.lower():
        state_dict["all_airports"] = True
        state_dict["multi_airport_prompt_active"] = False
        return flight_info_alt, assistant_response
    elif "done" in user_input.lower():
        state_dict["all_airports"] = False
        state_dict["multi_airport_prompt_active"] = False
        del state_dict["original_flight_info"]
        del state_dict["altered_flight_info"]
        assistant_response = "Enter another flight query..."
        return flight_info_alt, assistant_response

    # Parse the user input and update flight_info_alt with the selected airport code(s).
    origin_airport_codes = []
    destination_airport_codes = []
    all_airport_codes = []
    errors = []

    # Count how many valid selections are required and save airport codes of multi-airport cities only
    num_origin_airports = len(flight_info_alt["origin"]["airport"])
    num_destination_airports = len(flight_info_alt["destination"]["airport"])
    required_selections = 0
    if num_origin_airports > 1:
        required_selections += 1
        origin_airport_codes = [
            airport["code"] for airport in flight_info_alt["origin"]["airport"]
        ]
        all_airport_codes.extend(origin_airport_codes)
    if num_destination_airports > 1:
        required_selections += 1
        destination_airport_codes = [
            airport["code"] for airport in flight_info_alt["destination"]["airport"]
        ]
        all_airport_codes.extend(destination_airport_codes)

    num_all_airport_codes = len(all_airport_codes)

    if num_all_airport_codes < 10:
        # Check the input for 3-char alpha or 1-char numeric within the airports list range
        # Use a non-capturing group (?:...) for the OR condition, and escape curly braces within the f-string
        pattern = rf"\b(?:[A-Z]{{3}}|[1-{num_all_airport_codes}]{{1}})\b"
        valid_input = re.findall(pattern, user_input, re.IGNORECASE)
    else:
        # Currently n/a, this would be for future multi-leg flight requests
        # Check the input for 3-char alpha or 1 to 2-char numeric
        pattern = r"\b(?:[A-Z]{3}|[0-9]{1,2})\b"
        regex_matches = re.findall(pattern, user_input, re.IGNORECASE)
        # Remove invalid 1 to 2-char numerics if any
        valid_input = []
        for item in regex_matches:
            if item.isdigit():
                line_number = int(item)
                if line_number <= num_all_airport_codes and line_number != 0:
                    valid_input.append(item)
            else:
                valid_input.append(item)

    if not valid_input:
        if required_selections == 1:
            errors.append(
                "- Your input didn't include a 3-character airport code or valid line number."
            )
        else:
            errors.append(
                "- Your input didn't include any 3-character airport codes or valid line numbers."
            )

    else:
        if len(valid_input) != required_selections:
            if required_selections > 1:
                if len(valid_input) == 1:
                    errors.append(
                        f"- You provided 1 airport selection {valid_input} but {required_selections} are required"
                    )
                elif len(valid_input) > required_selections:
                    # Valid inputs are more than required
                    errors.append(
                        f"- You provided {len(valid_input)} airport selections {valid_input} but only {required_selections} are required"
                    )
                else:
                    # Valid inputs are less than required
                    errors.append(
                        f"- You provided {len(valid_input)} airport selections {valid_input} but {required_selections} are required"
                    )
            else:
                # Valid inputs must be > 1 since required inputs is 1 and they're not equal
                errors.append(
                    f"- You provided {len(valid_input)} airport selections {valid_input} but only 1 is required"
                )

        origin_airport_match = ""
        destination_airport_match = ""

        # Check if each string in valid_input matches an airport code in separate cities
        for item in valid_input:

            if item.isdigit():
                # Item is a line number
                line_number = int(item)

                if num_origin_airports > 1 and line_number <= num_origin_airports:
                    # Line number matches an origin airport
                    if origin_airport_match == "":
                        # Save the first match in case too many were provided
                        origin_airport_match += origin_airport_codes[line_number - 1]
                    else:
                        # More than 1 origin airport provided
                        city = flight_info_alt["origin"]["city"]
                        if origin_airport_match == all_airport_codes[line_number - 1]:
                            errors.append(
                                f"- You already selected airport '{origin_airport_match}' for {city}, so repeating line number '{item}' {choose_redundant_phrase()}."
                            )
                        else:
                            errors.append(
                                f"- You already selected airport '{origin_airport_match}' for {city}, so line number '{item}' {choose_redundant_phrase()}."
                            )
                else:
                    # Must be a destination line number
                    if required_selections == 1:
                        # Just 1 city with multi-airports
                        dest_index = line_number - 1
                    else:
                        # Both cities have multi-airports
                        dest_index = line_number - num_origin_airports - 1

                    if destination_airport_match == "":
                        # Save the first match in case too many were provided
                        destination_airport_match += destination_airport_codes[
                            dest_index
                        ]
                    else:
                        # More than 1 destination airport provided
                        city = flight_info_alt["destination"]["city"]
                        if (
                            destination_airport_match
                            == destination_airport_codes[dest_index]
                        ):
                            errors.append(
                                f"- You already selected airport '{destination_airport_match}' for {city}, so repeating line number '{item}' {choose_redundant_phrase()}."
                            )
                        else:
                            errors.append(
                                f"- You already selected airport '{destination_airport_match}' for {city}, so line number '{item}' {choose_redundant_phrase()}."
                            )

            else:
                # Item is 3-char alpha
                airport_code = item.upper()

                if airport_code not in all_airport_codes:
                    errors.append(
                        f"- Selected airport '{airport_code}' isn't one of the choices listed."
                    )
                else:
                    # Provided airport code matches one in multi-airport list
                    if airport_code in origin_airport_codes:
                        # Provided airport code matches an origin airport
                        if origin_airport_match == "":
                            # Save the first match in case too many were provided
                            origin_airport_match += airport_code
                        else:
                            # More than 1 origin airport provided
                            city = flight_info_alt["origin"]["city"]
                            if origin_airport_match == airport_code:
                                errors.append(
                                    f"- You already selected airport '{origin_airport_match}' for {city}, so entering '{airport_code}' again {choose_redundant_phrase()}."
                                )
                            else:
                                errors.append(
                                    f"- You already selected airport '{origin_airport_match}' for {city}, so airport code '{airport_code}' {choose_redundant_phrase()}."
                                )
                    else:
                        # Must be a destination airport code
                        if destination_airport_match == "":
                            # Save the first match in case too many were provided
                            destination_airport_match += airport_code
                        else:
                            # More than 1 destination airport provided
                            city = flight_info_alt["destination"]["city"]
                            if destination_airport_match == airport_code:
                                errors.append(
                                    f"- You already selected airport '{destination_airport_match}' for {city}, so entering '{airport_code}' again {choose_redundant_phrase()}."
                                )
                            else:
                                errors.append(
                                    f"- You already selected airport '{destination_airport_match}' for {city}, so airport code '{airport_code}' {choose_redundant_phrase()}."
                                )
        # End of for loop

    if errors:
        # Update assistant_response with error messages and original multi-airports list for redisplay
        if len(errors) > 1:
            header = "\nSorry, your selection had some issues:"
        else:
            header = "\nSorry, your selection had an issue:"
        errors.insert(0, header)
        # Add a blank line after the last error message
        errors.append("")
        if required_selections == 1:
            errors.append(
                "Please try again using one airport code or line number from the list above."
            )
        else:
            errors.append(
                f"Please try again using {required_selections} unique airport codes or line numbers from the list above."
            )
        errors.append(
            "Otherwise, type 'all' for all airports or 'done' to start a new query."
        )
        # Use \n in .join for markdown text
        assistant_response = (
            "_" + multi_airport_display_string + "_\n" + "\n".join(errors)
        )

    else:
        # Update flight_info_alt with the selected airport code(s)
        updated_origin_airports = [
            airport
            for airport in flight_info_alt["origin"]["airport"]
            if origin_airport_match == airport["code"]
        ]
        updated_destination_airports = [
            airport
            for airport in flight_info_alt["destination"]["airport"]
            if destination_airport_match == airport["code"]
        ]

        if updated_origin_airports:
            flight_info_alt["origin"]["airport"] = updated_origin_airports
        if updated_destination_airports:
            flight_info_alt["destination"]["airport"] = updated_destination_airports

    return flight_info_alt, assistant_response


###########################
#   Sort Flight Options   #
###########################
def sort_flight_options(flight_options, sort_button):
    """
    Sorts the flight options based on the selected sorting button.

    Args:
        flight_options (list): A list of flight options.
        sort_button (str): The selected sorting button.

    Returns:
        sorted_options (list): The sorted list of flight options.
    """
    if sort_button == "Price":
        # Sort by price (ascending) then by time
        return sorted(flight_options, key=lambda x: (float(x["price"]), x["time"]))
    elif sort_button == "Time":
        # Sort by time (ascending) then by price
        return sorted(flight_options, key=lambda x: (x["time"], float(x["price"])))
    else:
        # Sort by stops
        def get_sort_key(option):
            """Helper function to determine the sorting key."""
            # Extract the number of stops from the display text (duration, X stop/stops)
            stops_match = re.search(r"\(.*,\s*(\d+)\s*stop", option["display_text"])
            num_stops = (
                int(stops_match.group(1)) if stops_match else 0
            )  # Default to 0 for nonstops

            if sort_button == "Price (Nonstops first)":
                # Sort by number of stops (ascending), then price (ascending), then time
                return (num_stops, float(option["price"]), option["time"])

            elif sort_button == "Time (Nonstops first)":
                # Sort by number of stops (ascending), then time (ascending), then price
                return (num_stops, option["time"], float(option["price"]))

        return sorted(flight_options, key=get_sort_key)


##################################
#   Build Flight Options Batch   #
##################################
def build_flight_options_batch(sorted_options, start_index, batch_size=10):
    """
    Builds a formatted string for a batch of flight options.

    Args:
        sorted_options (list): The list of sorted flight options.
        start_index (int): The starting index for the batch.
        batch_size (int): The number of options in the batch.

    Returns:
        str: A formatted string containing a batch of flight options,
             or an empty string if no options.
    """
    response_lines = []

    for i in range(batch_size):
        list_index = i + start_index
        if list_index < len(sorted_options):
            response_lines.append(f"<br>--- Option {list_index + 1} ---")
            response_lines.append(f"{sorted_options[list_index]['display_text']}")

    if i + start_index + 1 >= len(sorted_options):
        response_lines.append(f"<br>--- End of flight options ---")
    else:
        response_lines.append(f"<br>--- View more flight options ---")

    return "<br>".join(response_lines)


################################
#   Show More Flight Options   #
################################
def show_more_flight_options(chat_history, state_dict, sort_button):
    """
    Builds the response for showing more flight options.

    Args:
        chat_history (list): The history of the conversation (list of dictionaries).
        state (dict or None): The current state dictionary.
        sort_button (str): The selected sorting button.

    Returns:
        chat_history (list): The updated chat history (list of dictionaries).
        state_dict (dict): The updated state dictionary.
    """
    response_lines = []

    if "flight_options_batch_n" in state_dict:
        # Get stored sort preference, else default to Price
        current_sort_preference = state_dict.get("active_sort_preference", "Price")

        if current_sort_preference != sort_button:
            # Sorting has changed, reset to the first batch and re-sort the list
            state_dict["flight_options_batch_n"] = 1
            state_dict["active_sort_preference"] = sort_button

            sorted_options = sort_flight_options(
                state_dict["flight_options_list"], sort_button
            )
            # Store the newly sorted list
            state_dict["flight_options_list"] = sorted_options

            response_lines.append(
                f"<br>--- Flight Options (sorted by {sort_button}) ---"
            )
            response_lines.append(state_dict["header_string"])

        else:
            # Sorting hasn't changed, just update the starting index
            state_dict["flight_options_batch_n"] += 1

        next_starting_index = (state_dict["flight_options_batch_n"] - 1) * 10

        if next_starting_index < len(state_dict["flight_options_list"]):
            response_lines.append(
                build_flight_options_batch(
                    state_dict["flight_options_list"], next_starting_index
                )
            )
            # Join the lines into a single string
            assistant_response = "<br>".join(response_lines)
            # Add the assistant's message to chat_history
            chat_history.append({"role": "assistant", "content": assistant_response})

        else:
            # No more options to display
            if (
                chat_history
                and chat_history[-1]["content"] != "<br>No more flights to display."
            ):  # Avoid adding the same message repeatedly
                chat_history.append(
                    {"role": "assistant", "content": "<br>No more flights to display."}
                )

    else:
        # No flight options were stored in state_dict (e.g., original search failed)
        if (
            chat_history
            and chat_history[-1]["content"]
            != "<br>No flight options available to display."
        ):  # Avoid adding the same message repeatedly
            chat_history.append(
                {
                    "role": "assistant",
                    "content": "<br>No flight options available to display.",
                }
            )

    return chat_history, state_dict


####################################
#   Chat Flight Assistant driver   #
####################################
def chat_flight_assistant(user_input, chat_history, state_dict, sort_button):
    """
    Processes user input for the flight assistant chatbot, handling conversational turns.

    Args:
        user_input (str): The user's message.
        chat_history (list): The history of the conversation (list of dictionaries).
                             e.g. [{user_message}, {bot_message}]
        state (dict or None): The current state, which will store the flight_info dictionary.
        sort_button (str): The selected sorting option.

    Returns:
        chat_history (list): The updated chat history (list of dictionaries).
        state_dict (dict): The updated state dictionary.
        clear_input (str): An empty string to clear the input textbox.
    """
    # Add the user's message to chat history
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_input})

    assistant_response = ""  # Initialize response string

    if (
        "multi_airport_prompt_active" in state_dict
        and state_dict["multi_airport_prompt_active"]
    ):
        # This is a response to the multi-airport question.
        flight_info_alt = state_dict["altered_flight_info"]
        multi_airport_display_string = state_dict["multi_airport_display_string"]
        flight_info, assistant_response = parse_multi_airport_response(
            user_input, state_dict, flight_info_alt, multi_airport_display_string
        )
        # Check for error response
        if assistant_response:
            # Add the assistant's message to chat_history
            chat_history.append({"role": "assistant", "content": assistant_response})
            return chat_history, state_dict, ""  # Return empty string to clear input

        # Multi-airports were eliminated in flight_info
        else:
            intent = "booking"

    # This is a new query
    else:
        state_dict["multi_airport_prompt_active"] = False
        state_dict["all_airports"] = False
        response = question_classifier(user_input)
        intent = response[0]["label"]
        flight_info, assistant_response = extract_flight_info(user_input, intent)

    if assistant_response:
        # Add the assistant's message to chat_history
        chat_history.append({"role": "assistant", "content": assistant_response})
        return chat_history, state_dict, ""  # Return empty string to clear input

    #-----------------------#
    #   Status processing   #
    #-----------------------#
    if intent == "status":

        if flight_info["airline_code"] is None:
            if flight_info["flight_number"] is None:
                assistant_response = "Sorry, I couldn't find a valid airline name/code or flight number in your status request."
                assistant_response += f"\n\n{choose_retry_phrase()}"
            else:
                assistant_response = "Sorry, I couldn't find a valid airline name/code in your status request."
                assistant_response += f"\n\n{choose_retry_phrase()}"
        elif flight_info["flight_number"] is None:
            assistant_response = (
                "Sorry, I couldn't find a flight number in your status request."
            )
            assistant_response += f"\n\n{choose_retry_phrase()}"
        else:
            flight_ident = flight_info["airline_code"] + flight_info["flight_number"]
            flight_data = get_flight_status(flight_ident)
            assistant_response = build_flight_status_response(flight_data, flight_ident)

    #------------------------#
    #   Booking processing   #
    #------------------------#
    elif intent == "booking":

        response_lines = []  # Use a list to build the response string

        # Parsing errors for booking queries were already processed in extract_flight_info.
        # If here, all flight_info keys will have values.
        requested_date = parse_query_date(flight_info["date"])
        requested_cabin = flight_info["cabin"]

        if requested_date:
            origin_airports = flight_info["origin"].get("airport", [])
            destination_airports = flight_info["destination"].get("airport", [])

            # Check if origin and/or destination cities have multiple airports
            if state_dict["all_airports"] == False and (
                len(origin_airports) > 1 or len(destination_airports) > 1
            ):

                # Store the flight info in state_dict for a follow-up user reply
                state_dict["altered_flight_info"] = flight_info
                # Save all multi-airports once in case the user wants to retry with different airports
                if state_dict["multi_airport_prompt_active"] == False:
                    state_dict["original_flight_info"] = copy.deepcopy(flight_info)
                    state_dict["multi_airport_prompt_active"] = True

                # Ask the user for clarification and list options
                # (header will be inserted after updating state_dict)
                origin_code = None
                destination_code = None

                current_line_number = 0
                if len(origin_airports) > 1:
                    origin_code = flight_info["origin"]["airport"][0]["code"]
                    response_lines.append(
                        f"For origin '{flight_info['origin']['city']}', multiple airports were found:"
                    )
                    for i, airport in enumerate(origin_airports):
                        # Display the entire string from airport_codes for key ['airport code']
                        # e.g. "SPI": "Springfield, IL: Abraham Lincoln Capital",
                        city_airport_str = (
                            airport_codes[airport["code"]] + f" ({airport['code']})"
                        )
                        response_lines.append(f"  {i+1}. {city_airport_str}")
                        current_line_number += 1
                    response_lines.append("")

                if len(destination_airports) > 1:
                    destination_code = flight_info["destination"]["airport"][0]["code"]
                    response_lines.append(
                        f"For destination '{flight_info['destination']['city']}', multiple airports were found:"
                    )
                    for i, airport in enumerate(
                        destination_airports, start=current_line_number
                    ):
                        # Display the entire string from airport_codes for key ['airport code']
                        # e.g. "SGF": "Springfield, MO: Springfield-Branson National",
                        city_airport_str = (
                            airport_codes[airport["code"]] + f" ({airport['code']})"
                        )
                        response_lines.append(f"  {i+1}. {city_airport_str}")
                    response_lines.append("")

                if origin_code is not None and destination_code is not None:
                    first_destination_line_number = len(origin_airports) + 1
                    response_lines.append(
                        f"Please specify which airports you'd like to use (e.g., 1,{first_destination_line_number} or {origin_code}-{destination_code})."
                    )
                elif origin_code is not None:
                    response_lines.append(
                        f"Please specify which airport you'd like to use (e.g., 1 or {origin_code})."
                    )
                else:
                    response_lines.append(
                        f"Please specify which airport you'd like to use (e.g., 1 or {destination_code})."
                    )

                # Join the lines into a single string
                state_dict["multi_airport_display_string"] = "<br>".join(response_lines)

                # Add trailer and header first time only to avoid redisplaing them when flights aren't found
                response_lines.append(
                    "Otherwise, type 'all' for all airports or 'done' to start a new query."
                )
                response_lines.insert(0, "Multiple airports exist for your search.\n")

            # Case where exactly one airport is found for each city,
            # or 'all' was selected for multi-airports
            elif origin_airports and destination_airports:
                origin_code = origin_airports[0]["code"]
                destination_code = destination_airports[0]["code"]
                passengers = {}
                passengers["adult"] = 1
                passengers["senior"] = 0
                passengers["childrenAges"] = [0]
                passengers["infantInLap"] = 0
                passengers["infantInSeat"] = 0

                booking_data = get_flights_to_book(flight_info, passengers, state_dict)

                header_str, options_list = build_booking_response(
                    booking_data, requested_date, timezone_finder
                )

                # No flights were found
                if "No flights" in header_str:
                    if state_dict["multi_airport_prompt_active"] == True:
                        # This was after a multi-airport selection
                        original_flight_info = state_dict["original_flight_info"]
                        origin_airports = original_flight_info["origin"]["airport"]
                        destination_airports = original_flight_info["destination"][
                            "airport"
                        ]
                        num_multi_airport_cities = 0
                        if len(origin_airports) > 1:
                            num_multi_airport_cities += 1
                        if len(destination_airports) > 1:
                            num_multi_airport_cities += 1
                        # Add follow-up display lines
                        response_lines.append(
                            "_" + state_dict["multi_airport_display_string"] + "_"
                        )
                        response_lines.append(
                            f"\nNo flights were found for the requested airports ({origin_code}-{destination_code}).\n"
                        )
                        if num_multi_airport_cities > 1:
                            response_lines.append(
                                "You can try again with different airports from the list above."
                            )
                        else:
                            response_lines.append(
                                "If you'd like to try again, select a different airport from the list above."
                            )
                        response_lines.append(
                            "Otherwise, type 'all' for all airports or 'done' to start a new query."
                        )

                        state_dict["altered_flight_info"] = copy.deepcopy(
                            state_dict["original_flight_info"]
                        )
                    else:
                        header_str = f"No flights were found for the requested airports ({origin_code}-{destination_code})."
                        response_lines.append(header_str)

                # Flights were found
                else:
                    # Sort options based on the selected button
                    sorted_options = sort_flight_options(options_list, sort_button)

                    # Store the booking info in state_dict for follow-up user requests
                    state_dict["multi_airport_prompt_active"] = False
                    state_dict["all_airports"] == False
                    state_dict["active_sort_preference"] = sort_button
                    state_dict["flight_options_batch_n"] = 1
                    state_dict["header_string"] = header_str
                    state_dict["flight_options_list"] = sorted_options

                    # Build the response string for the first 10 options
                    response_lines.append(
                        f"--- Flight Options (sorted by {sort_button}) ---"
                    )
                    response_lines.append(header_str)
                    response_lines.append(build_flight_options_batch(sorted_options, 0))

            else:
                # Shouldn't get here since origin, destination errors were checked before building flight_info
                response_lines.append(
                    "I had a problem extracting origin, destination airports. Could you retry?"
                )

        else:
            # The requested date couldn't be converted to a datetime object
            response_lines.append(
                f"The requested date '{flight_info['date']}' is invalid. Please recheck and try again."
            )

        # Join the lines into a single string
        assistant_response = "<br>".join(response_lines)

    #------------------------#
    #   General processing   #
    #------------------------#
    elif intent == "general":
        # Create lists of varied responses
        responses_to_general_questions = [
            "I apologize, I don't have that information. I can help with flight status or booking requests if you'd like.",
            "I can only assist with flight status and booking at the moment. How else can I help with your travel?",
            "My focus is on flight information. If you have a status or booking query, I'd be happy to help!",
            "I'm not equipped to handle that type of request just yet. I can provide flight status updates or help you find flights.",
            "That's outside my current capabilities. I specialize in looking up flight statuses and helping with bookings.",
            "For now, I can only discuss flight status and booking. Please ask me about a flight!",
        ]
        responses_to_general_input = [
            "I'm sorry, I didn't understand that. If you make a flight status or booking request, I'm here to help.",
            "I'm having trouble understanding your request. Could you please rephrase or tell me if you're looking for flight status or trying to book?",
            "Could you clarify your request? I can help with flight status or finding flights.",
            "My apologies, I didn't quite catch that. Are you asking about a flight's status or looking to book a flight?",
            "It seems there was a communication issue. Please tell me if you need a flight update or want to search for flights.",
            "I'm not sure what you meant. Please try again, focusing on flight status or booking.",
        ]
        # Choose a random response
        if "?" in user_input:
            assistant_response = random.choice(responses_to_general_questions)
        else:
            assistant_response = random.choice(responses_to_general_input)

    # If no specific response was generated, provide a default
    if not assistant_response.strip():
        assistant_response = (
            "I'm sorry, I couldn't process that request. Can you please rephrase?"
        )

    # Add the assistant's message to chat_history
    chat_history.append({"role": "assistant", "content": assistant_response})

    return chat_history, state_dict, ""  # Return empty string to clear input


##################
#   Clear Chat   #
##################
def clear_chat(chat_history, state_dict):
    """
    Clears the chat history and resets the state dictionary.

    Args:
        chat_history (list): The current chat history.
        state_dict (dict): The current state dictionary.

    Returns:
        list: An empty list for the chat history.
        dict: An empty dictionary for the state dictionary.
        str: An empty string to clear the input textbox.
    """
    return [], {}, ""


##################
### Gradio app ###
##################

# Create the Chat Interface
with gr.Blocks() as chat_interface:

    # Add a title and description using Markdown
    gr.Markdown(
        """
        # Flight Assistant Chat Demo
        Ask about flight status or book a flight in a conversational style.
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            # Instantiate the Chatbot component.
            chatbot = gr.Chatbot(
                label="Flight Assistant",  # Label for the component.
                elem_id="chatbot",
                type="messages",
            )

            # Instantiate the Textbox for user input.
            user_input_textbox = gr.Textbox(
                placeholder="Enter your flight query or response...",
                container=False,
                scale=7,
            )

        with gr.Column(scale=1):
            sort_buttons = gr.Radio(
                ["Price", "Price (Nonstops first)", "Time", "Time (Nonstops first)"],
                value="Price",
                label="Sort by:",
            )
            show_more_flights_button = gr.Button("Show More Flight Options")
            dummy_button = gr.Button("")
            clear_button = gr.Button("Clear Chat")

    state_dict = gr.State(value={})

    # Create an event listener for the input textbox.
    # When the user submits a message via the Textbox (`user_input_textbox.submit`):
    #   - Call the `chat_flight_assistant` function.
    #   - Pass the Textbox content, Chatbot history, state, and sort buttons as inputs.
    #   - Update the Chatbot, state, and clear the Textbox with the function's outputs.
    user_input_textbox.submit(
        fn=chat_flight_assistant,
        inputs=[user_input_textbox, chatbot, state_dict, sort_buttons],
        outputs=[
            chatbot,
            state_dict,
            user_input_textbox,
        ],  # Add user_input_textbox as output to clear it
    )

    # Create an event listener for the "Show more flight options" button.
    # When the button is clicked (`show_more_flights_button.click`):
    #   - Call the `show_more_flight_options` function.
    #   - Pass the Chatbot history, state, and selected sort button value as inputs.
    #   - Update the Chatbot, state, and clear the Textbox with the function's outputs.
    show_more_flights_button.click(
        fn=show_more_flight_options,
        inputs=[chatbot, state_dict, sort_buttons],  # Pass chat history from chatbot
        outputs=[chatbot, state_dict],  # Update chatbot with new history
    )

    # Create an event listener for the Clear button.
    # When the button is clicked (`clear_button.click`):
    #   - Call the `clear_chat` function.
    #   - Pass the Chatbot history and state as inputs.
    #   - Update the Chatbot, state, and clear the Textbox with the function's outputs.
    clear_button.click(
        fn=clear_chat,
        inputs=[chatbot, state_dict],  # Pass chat_history and state
        outputs=[
            chatbot,
            state_dict,
            user_input_textbox,
        ],  # Update chatbot, state, and input textbox
    )


### Launch the app ###
chat_interface.launch()
