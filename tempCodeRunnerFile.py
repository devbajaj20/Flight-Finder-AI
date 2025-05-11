import google.generativeai as genai
from dotenv import load_dotenv
import os
from GetFlights import GetFlightData
import pandas as pd
import chainlit as cl
import asyncio
import logging
logging.basicConfig(level=logging.DEBUG)


load_dotenv()
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
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

def GetFlightDetails(*, 
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

        departure_date: A list of departure dates in the format 'YYYY-MM-DD'. The year would be always 2025. For example: ['2025-01-02']. If the user mentions additional dates, append them to the list (e.g., ['2025-01-02', '2025-03-21']). If the user has no preference, send an empty list ([]).

        arrival_date: A list of arrival dates in the format 'YYYY-MM-DD'.The year would be always 2025. For example: ['2025-01-02']. If the user mentions additional dates, append them to the list (e.g., ['2025-01-02', '2025-03-21']). If the user has no preference, send an empty list ([]).

        price: A list specifying price ranges for flights.
            If the user mentions a price greater than 3000, use the format [3000, 1e24].
            If the user mentions a price less than 3000, use the format [0, 3000].
            If the user mentions a range (e.g., more than 2000 and less than 5000), use the format [2000, 5000].
            If the user specifies prices for additional flights, append them as separate lists (e.g., [[3000, 1e24], [2000, 5000]]).
            If the user has no preference, send an empty list ([]).
            
        airline: A list of preferred airlines. For example: ['IndiGo']. If the user mentions additional airlines, append them to the list (e.g., ['Indigo', 'Air India']). If the user has no preference, send an empty list ([]). Available flights are: ["Air India", "SpiceJet", "IndiGo", "Vistara", "GoAir", "AirAsia"]

        duration: A list specifying flight durations in minutes. If the user mentions durations in hours, convert them to minutes (1 hour = 60 minutes).
            For durations more than 3h, use the format [180, 1e24].
            For durations less than 2h, use the format [0, 120].
            For durations within a range (e.g., more than 2h and less than 5h), use the list format [120, 300].
            If the user specifies durations for additional flights, append them as separate tuples in list (e.g., [[180, 1e24], [120, 300]]).
            If the user has no preference, send an empty list ([]).
    Returns:
        dict: A dictionary containing all the flight details."""
        
    print(origin,destination,departure_date,arrival_date,price,airline,duration)
    flights = GetFlightData(origin=origin,destination=destination,departure_date=departure_date,arrival_date=arrival_date,price=price,airline=airline,duration=duration)
    
    if not flights:
        return  "No flight details found"
    
    else:
        data = pd.DataFrame(flights)
        print(data)
    
    return flights

async def CallGemini(query):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash",system_instruction=system_message, tools=[GetFlightDetails])
    chat_session = model.start_chat(history=[], enable_automatic_function_calling=True)
    response = chat_session.send_message(query)
    print(response.text)
    return response.text

@cl.on_chat_start
async def StartChat():
    welcome_message = "Hi there! ✈️ Ready to help you find the best flights. How can I assist you today?"
    await cl.Message(content=welcome_message, author="Answer").send()
    
@cl.on_message
async def SendMsg(user_message: cl.Message):
    response_text = await CallGemini(user_message.content)
    
    msg = cl.Message(content=" ")
    for chunk in response_text:
        await msg.stream_token(chunk)
        await asyncio.sleep(0.05)

    await msg.update()
    
    
GetFlightDetails()