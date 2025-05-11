import google.generativeai as genai
from dotenv import load_dotenv
import os
from flight_data_service import FlightDataService
import pandas as pd
import chainlit as cl
import asyncio
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Configure port for Railway
PORT = int(os.getenv('PORT', 8000))

# Initialize FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the flight data service
flight_service = FlightDataService()

# FastAPI routes
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Flight Finder AI is running"}

system_message="""
            You are a helpful technical flight support assistant. Your role is to assist users in finding flight details in a clear and easy-to-read format. When flight details are found, format the information in a readable table with the following columns:
            - FlightNumber
            - Origin
            - Destination
            - DepartureDate
            - DepartureTime
            - ArrivalDate
            - ArrivalTime
            - Price
            - Airline
            - Duration
            - Duration_min (in minutes)
            You should ensure that all the details are displayed properly, including correct formatting for time, dates, prices, and flight durations.
            Additionally, offer concise and helpful explanations or answers to any flight-related queries.
            Remember these are available flights:  ["Air India", "SpiceJet", "IndiGo", "Vistara", "GoAir", "AirAsia"]. Follow the flight names stricly. Do not make them uppercase or lowercase randomly.
            Remember these are available airports:  ["Bangalore", "Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Pune", "Jaipur"]. Follow the flight names stricly. Do not make them uppercase or lowercase randomly.
        """
from typing import List

async def GetFlightDetails(*, 
                     origin: List[str] = [], 
                     destination: List[str] = [], 
                     departure_date: List[str] = [], 
                     arrival_date: List[str] = [], 
                     price: List[float] = [], 
                     airline: List[str] = [], 
                     duration: List[int] = []) -> dict:
    """
    Get the details of a flight and show them in the form of table. This function processes user queries to extract flight details.

    Args:
        origin: A list of origin city/cities for the flight. For example: ['Kolkata']. If the user mentions additional origins, append them to the list (e.g., ['Kolkata', 'Delhi']). If the user has no preference, send an empty list ([]). Available airports are: ["Bangalore", "Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Pune", "Jaipur"]

        destination: A list of destination city/cities for the flight. For example: ['Chennai']. If the user mentions additional destinations, append them to the list (e.g., ['Chennai', 'Delhi']). If the user has no preference, send an empty list ([]). Available airports are: ["Bangalore", "Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Pune", "Jaipur"]

        departure_date: A list of departure dates in the format 'YYYY-MM-DD'. For example: ['2025-01-02']. If the user mentions additional dates, append them to the list (e.g., ['2025-01-02', '2025-03-21']). If the user has no preference, today's date will be used.

        arrival_date: A list of arrival dates in the format 'YYYY-MM-DD'. Not used in real-time search as we focus on departure dates.

        price: A list specifying price ranges for flights. Not used in real-time search as prices are dynamic.
            
        airline: A list of preferred airlines. For example: ['IndiGo']. If the user mentions additional airlines, append them to the list (e.g., ['Indigo', 'Air India']). If the user has no preference, send an empty list ([]). Available flights are: ["Air India", "SpiceJet", "IndiGo", "Vistara", "GoAir", "AirAsia"]

        duration: A list specifying flight durations in minutes. Not used in real-time search as durations are dynamic.
            
    Returns:
        dict: A dictionary containing all the flight details."""
    
    print(origin, destination, departure_date, arrival_date, price, airline, duration)
    
    # Use first origin and destination if provided, otherwise return empty
    origin_city = origin[0] if origin else None
    dest_city = destination[0] if destination else None
    
    # Use provided departure date or today's date
    search_date = departure_date[0] if departure_date else datetime.now().strftime('%Y-%m-%d')
    
    # Get real-time flight data
    flights = await flight_service.get_flights(
        origin=origin_city,
        destination=dest_city,
        departure_date=search_date
    )
    
    # Filter by airline if specified
    if airline:
        flights = [f for f in flights if f['Airline'] in airline]
    
    if not flights:
        return "No flight details found"
    
    data = pd.DataFrame(flights)
    print(data)
    
    return flights

async def CallGemini(query):
    try:
        import aiohttp
        import json
        from datetime import datetime
        
        # First try to get flight details
        try:
            # Extract date from query or use tomorrow's date
            if "tomorrow" in query.lower():
                date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                # Try to extract date from query
                import re
                date_pattern = r'\b(\d{1,2})\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})\b'
                date_match = re.search(date_pattern, query.lower())
                if date_match:
                    day = date_match.group(1)
                    year = date_match.group(2)
                    month = query.lower()[date_match.start():date_match.end()].split()[1]
                    month_map = {
                        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                    }
                    month = month_map.get(month[:3], '01')
                    date = f"{year}-{month}-{day.zfill(2)}"
                    logging.info(f"Extracted date from query: {date}")
                else:
                    date = datetime.now().date().strftime("%Y-%m-%d")
                    logging.info(f"Using default date: {date}")
            
            # Extract filters from query
            query_lower = query.lower()
            
            # Initialize flight_info variable at the start
            flight_info = ""
            
            # Extract origin and destination from all supported cities
            supported_cities = ["bangalore", "delhi", "mumbai", "chennai", "kolkata", "hyderabad", "pune", "jaipur"]
            
            # Find origin city
            origin = None
            for city in supported_cities:
                if f"from {city}" in query_lower or f"origin {city}" in query_lower:
                    origin = city.capitalize()
                    break
            
            # Find destination city
            destination = None
            for city in supported_cities:
                if f"to {city}" in query_lower or f"destination {city}" in query_lower:
                    destination = city.capitalize()
                    break
            
            # Validate origin and destination
            if not origin or not destination:
                return "Please specify both origin and destination cities. For example: 'Show me flights from Delhi to Mumbai'"
            
            # Extract airline filter
            selected_airline = None
            airlines = ["air india", "spicejet", "indigo", "vistara", "goair", "airasia"]
            for airline in airlines:
                if airline in query_lower:
                    selected_airline = airline.title()
                    break
            
            # Get flight details from Amadeus
            flight_results = await flight_service.get_flights(
                origin=origin,
                destination=destination,
                departure_date=date
            )
            
            if flight_results:
                # Apply filters
                filtered_flights = flight_results
                
                # Apply airline filter
                if selected_airline:
                    filtered_flights = [f for f in filtered_flights if f['Airline'].lower() == selected_airline.lower()]
                
                # Remove duplicate flights based on flight number and departure time
                seen_flights = set()
                unique_flights = []
                for flight in filtered_flights:
                    flight_key = (flight['FlightNumber'], flight['DepartureTime'])
                    if flight_key not in seen_flights:
                        seen_flights.add(flight_key)
                        unique_flights.append(flight)
                
                # Sort flights by departure time
                unique_flights.sort(key=lambda x: x['DepartureTime'])
                
                # Create table header with alignment
                flight_info = "\n| Airline | Flight | Departure Time | Duration | Price |\n"
                flight_info += "|:--------|:-------|:--------------|:---------|------:|\n"
                
                # Add each flight as a row
                for flight in unique_flights:
                    # Format time to be more readable
                    dep_time = f"{flight['DepartureTime'][:5]}"
                    
                    # Format price with thousand separator
                    price = "{:,.0f}".format(flight['Price'])
                    
                    # Add formatted row to flight info
                    flight_info += f"| {flight['Airline']} | {flight['FlightNumber']} | {dep_time} | {flight['Duration']} | ₹{price} |\n"
                
                # Build filter message
                filters = []
                if selected_airline:
                    filters.append(f"for {selected_airline}")
                
                filter_msg = " and ".join(filters)
                flight_info += f"\nShowing {len(unique_flights)} flights {filter_msg} (sorted by departure time)"
                
                query = f"Here are the available flights from {origin} to {destination} for {date}:\n{flight_info}\n\nNotes:\n- All times are in 24-hour format\n- Prices are in Indian Rupees (₹)\n- Duration shows hours and minutes\n\nWould you like to:\n1. Filter by airline\n2. Sort by price\n3. Sort by departure time\n4. Search for different dates\n\nPlease let me know how I can help you further."
            
        except Exception as e:
            logging.error(f"Error getting flight details: {e}")
            query = f"Error getting flight details: {str(e)}. " + query
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json"
        }
        
        # Create the prompt with context
        prompt = f"""You are a flight booking assistant. Present the flight information EXACTLY as shown below, preserving all markdown formatting.

Here are the available flights from {origin} to {destination} for {date}:

{flight_info}

Notes:
- All times are in 24-hour format
- Prices are in Indian Rupees (₹)
- Duration shows hours and minutes

Would you like to:
1. Filter by airline
2. Sort by price
3. Sort by departure time
4. Search for different dates

Please let me know how I can help you further."""
        
        if not flight_results:
            prompt = """I apologize, but I couldn't find any flights matching your criteria at the moment. Here are some suggestions:

1. Try different dates: Flight availability can vary by day
2. Check alternative airlines: Different carriers may have other options
3. Visit these resources for more options:
   - Popular flight booking websites (MakeMyTrip, Cleartrip)
   - Airline websites (Air India, IndiGo, SpiceJet)
   - Local travel agencies

Would you like to try searching for a different date or route?"""
        
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 1024,
                "stopSequences": ["tool_code", "tool_code_output"]
            }
        }
        
        params = {
            "key": "AIzaSyDM_a8Cp6_m9MFk_8H0e8xaBMbHbMK4tgs"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, params=params, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logging.info(f"API Response: {result}")
                    if 'candidates' in result and result['candidates']:
                        return result['candidates'][0]['content']['parts'][0]['text']
                    else:
                        return "I apologize, but I couldn't find any flight information at the moment."
                else:
                    error_text = await response.text()
                    logging.error(f"API Error: {error_text}")
                    return f"I apologize, but I encountered an error while searching for flights. Please try again later."
                    
    except Exception as e:
        logging.error(f"Error in CallGemini: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

@cl.on_chat_start
async def StartChat():
    welcome_message = "Hi there! ✈️ Ready to help you find the best flights. How can I assist you today?"
    await cl.Message(content=welcome_message, author="Answer").send()
    
@cl.on_message
async def SendMsg(user_message: cl.Message):
    try:
        # Send a thinking message
        msg = cl.Message(content="Searching for flights...")
        await msg.send()

        # Check if this is a filter/sort request
        query_lower = user_message.content.lower()
        
        if "filter by airline" in query_lower:
            # Get list of available airlines
            airlines = ["Air India", "SpiceJet", "IndiGo", "Vistara", "GoAir", "AirAsia"]
            response = "Please select an airline to filter by:\n\n"
            for i, airline in enumerate(airlines, 1):
                response += f"{i}. {airline}\n"
            await cl.Message(content=response).send()
            return
            
        elif "sort by price" in query_lower:
            # Get the current flights and sort by price
            response_text = await CallGemini(user_message.content)
            if "No flights found" not in response_text:
                # Extract the table from the response
                table_start = response_text.find("| Airline")
                table_end = response_text.find("\n\nNotes:")
                if table_start != -1 and table_end != -1:
                    table = response_text[table_start:table_end].strip()
                    # Parse the table and sort by price
                    lines = table.split('\n')
                    header = lines[0:2]
                    data_lines = lines[2:]
                    # Sort by price (last column)
                    sorted_lines = sorted(data_lines, key=lambda x: float(x.split('|')[-2].strip().replace('₹', '').replace(',', '')))
                    # Reconstruct the response
                    sorted_table = '\n'.join(header + sorted_lines)
                    response_text = response_text[:table_start] + sorted_table + response_text[table_end:]
                    response_text = response_text.replace("sorted by departure time", "sorted by price")
            await cl.Message(content=response_text).send()
            return
            
        elif "sort by departure time" in query_lower:
            # Get the current flights and sort by departure time
            response_text = await CallGemini(user_message.content)
            if "No flights found" not in response_text:
                # Extract the table from the response
                table_start = response_text.find("| Airline")
                table_end = response_text.find("\n\nNotes:")
                if table_start != -1 and table_end != -1:
                    table = response_text[table_start:table_end].strip()
                    # Parse the table and sort by departure time
                    lines = table.split('\n')
                    header = lines[0:2]
                    data_lines = lines[2:]
                    # Sort by departure time (third column)
                    sorted_lines = sorted(data_lines, key=lambda x: x.split('|')[3].strip())
                    # Reconstruct the response
                    sorted_table = '\n'.join(header + sorted_lines)
                    response_text = response_text[:table_start] + sorted_table + response_text[table_end:]
            await cl.Message(content=response_text).send()
            return
            
        elif "search for different dates" in query_lower:
            response = "Please enter a new date in the format 'DD Month YYYY' (e.g., '27 May 2025')"
            await cl.Message(content=response).send()
            return

        # Get the response for normal flight search
        response_text = await CallGemini(user_message.content)
        
        # Send the final response as a new message
        await cl.Message(content=response_text).send()

    except Exception as e:
        logging.error(f"Error in SendMsg: {str(e)}")
        await cl.Message(content=f"I apologize, but I encountered an error: {str(e)}").send()