##Attributes
# Flight Number: Unique identifier for each flight.
# Origin: The starting point of the flight (e.g., Bangalore).
# Destination: The endpoint of the flight (e.g., Delhi).
# Departure Time: When the flight leaves the origin.
# Arrival Time: When the flight reaches the destination.
# Price: The cost of the flight.
# Airline: The name of the airline operating the flight.
# Duration: Total flight time.

import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

faker = Faker()


airports = ["Bangalore", "Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Pune", "Jaipur"]
airlines = ["Air India", "SpiceJet", "IndiGo", "Vistara", "GoAir", "AirAsia"]

def generate_flight_dataframe(num_records=100):
    flight_data = {
        "FlightNumber": [],
        "Origin": [],
        "Destination": [],
        "DepartureDate": [],
        "DepartureTime": [],
        "ArrivalDate": [],
        "ArrivalTime": [],
        "Price": [],
        "Airline": [],
        "Duration": [],
        "Duration_min": []
    }

   
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    end_date = datetime(2025, 12, 31, 23, 59, 59)

    for _ in range(num_records):
        origin = random.choice(airports)
        destination = random.choice([a for a in airports if a != origin])  
        departure_datetime = faker.date_time_between(start_date=start_date, end_date=end_date)
        duration_minutes = random.randint(60, 240)  
        arrival_datetime = departure_datetime + timedelta(minutes=duration_minutes)
        price = random.randint(3000, 10000)  
        airline = random.choice(airlines)
        flight_number = faker.bothify(text="??###") 

        departure_date = departure_datetime.date()
        departure_time = departure_datetime.time()
        arrival_date = arrival_datetime.date()
        arrival_time = arrival_datetime.time()

        flight_data["FlightNumber"].append(flight_number)
        flight_data["Origin"].append(origin)
        flight_data["Destination"].append(destination)
        flight_data["DepartureDate"].append(departure_date)
        flight_data["DepartureTime"].append(departure_time)
        flight_data["ArrivalDate"].append(arrival_date)
        flight_data["ArrivalTime"].append(arrival_time)
        flight_data["Price"].append(price)
        flight_data["Airline"].append(airline)
        flight_data["Duration"].append(f"{duration_minutes // 60}h {duration_minutes % 60}m")
        flight_data["Duration_min"].append(duration_minutes)
    
    return pd.DataFrame(flight_data)

flight_df = generate_flight_dataframe(100)

# flight_df.to_csv("flights_2025.csv", index=False)


##To check if data is correctly inserted
import sqlite3

connection = sqlite3.connect("FlightData.db")
flight_df.to_sql('FlightData',connection,if_exists="replace",index=False)

result = pd.read_sql('SELECT * From FlightData',connection)

print(result)
connection.close()

