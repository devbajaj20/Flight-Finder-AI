import sqlite3

def GetFlightData(*, 
                  origin=[], 
                  destination=[], 
                  departure_date=[], 
                  arrival_date=[], 
                  price=[], 
                  airline=[], 
                  duration=[]):
    connection = sqlite3.connect("FlightData.db")
    cursor = connection.cursor()
    
    query = "SELECT * FROM FlightData WHERE 1=1"
    
    if origin:
        origins_placeholder = ', '.join(['?'] * len(origin))  
        query += f" AND Origin IN ({origins_placeholder})"
 
    if destination:
        destinations_placeholder = ', '.join(['?'] * len(destination))  
        query += f" AND Destination IN ({destinations_placeholder})"
    

    if departure_date:
        departure_placeholder = ', '.join(['?'] * len(departure_date))  
        query += f" AND DepartureDate IN ({departure_placeholder})"
  
    if arrival_date:
        arrival_placeholder = ', '.join(['?'] * len(arrival_date))  
        query += f" AND ArrivalDate IN ({arrival_placeholder})"
    
    if price:
        query += f" AND Price BETWEEN ? AND ?"
    
    if airline:
        airlines_placeholder = ', '.join(['?'] * len(airline))  
        query += f" AND Airline IN ({airlines_placeholder})"

    if duration:
        query += f" AND Duration_min BETWEEN ? AND ?"
   
    values = []
    
    if origin:
        values.extend(origin)
    if destination:
        values.extend(destination)
    if departure_date:
        values.extend(departure_date)
    if arrival_date:
        values.extend(arrival_date)
    if price:
        values.extend([price[0][0], price[0][1]])  
    if airline:
        values.extend(airline)
    if duration:
        values.extend([duration[0][0], duration[0][1]])  

    cursor.execute(query, values)
    
    rows = cursor.fetchall()
    
    connection.close()   

    flight_details = {
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
    
    if rows:
        for row in rows:
            flight_details["FlightNumber"].append(row[0])
            flight_details["Origin"].append(row[1])
            flight_details["Destination"].append(row[2])
            flight_details["DepartureDate"].append(row[3])
            flight_details["DepartureTime"].append(row[4])
            flight_details["ArrivalDate"].append(row[5])
            flight_details["ArrivalTime"].append(row[6])
            flight_details["Price"].append(row[7])
            flight_details["Airline"].append(row[8])
            flight_details["Duration"].append(row[9])
            flight_details["Duration_min"].append(row[10])
        
        return flight_details

    


