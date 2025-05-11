from amadeus import Client, ResponseError
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging

class FlightDataService:
    def __init__(self):
        load_dotenv()
        self.amadeus = Client(
            client_id=os.getenv('AMADEUS_API_KEY'),
            client_secret=os.getenv('AMADEUS_API_SECRET')
        )
        
        # IATA code mapping for our supported cities
        self.city_to_iata = {
            'Bangalore': 'BLR',
            'Delhi': 'DEL',
            'Mumbai': 'BOM',
            'Chennai': 'MAA',
            'Kolkata': 'CCU',
            'Hyderabad': 'HYD',
            'Pune': 'PNQ',
            'Jaipur': 'JAI'
        }
        
        # Airline code mapping
        self.airline_codes = {
            'AI': 'Air India',
            'SG': 'SpiceJet',
            '6E': 'IndiGo',
            'UK': 'Vistara',
            'G8': 'GoAir',
            'I5': 'AirAsia',
            'QP': 'Akasa Air',
            '9I': 'Alliance Air',
            'IX': 'Air India Express',
            'S5': 'Star Air',
            'ZO': 'Zoom Air',
            'TR': 'TruJet'
        }

    def _convert_to_iata(self, city):
        return self.city_to_iata.get(city)

    def _get_airline_name(self, code):
        return self.airline_codes.get(code, code)

    def _format_duration(self, duration_str):
        # Convert PT2H30M format to hours and minutes
        hours = 0
        minutes = 0
        if 'H' in duration_str:
            hours = int(duration_str.split('H')[0].replace('PT', ''))
            if 'M' in duration_str:
                minutes = int(duration_str.split('H')[1].replace('M', ''))
        elif 'M' in duration_str:
            minutes = int(duration_str.replace('PT', '').replace('M', ''))
        
        total_minutes = hours * 60 + minutes
        return f"{hours}h {minutes}m", total_minutes

    async def get_flights(self, origin=None, destination=None, departure_date=None):
        try:
            logging.info(f"Searching flights from {origin} to {destination} on {departure_date}")
            
            if not origin or not destination or not departure_date:
                logging.warning("Missing required parameters")
                return []

            origin_iata = self._convert_to_iata(origin)
            destination_iata = self._convert_to_iata(destination)
            
            if not origin_iata or not destination_iata:
                logging.error(f"Invalid city code - Origin: {origin} ({origin_iata}), Destination: {destination} ({destination_iata})")
                raise Exception(f"Could not find airport codes for {origin} to {destination}. Please check city names.")

            logging.info(f"Searching flights with IATA codes: {origin_iata} to {destination_iata}")
            response = self.amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin_iata,
                destinationLocationCode=destination_iata,
                departureDate=departure_date,
                adults=1,
                max=100,
                currencyCode="INR"
            )

            if not response.data:
                logging.info(f"No flights found for {origin} to {destination} on {departure_date}")
                return []

            logging.info(f"Found {len(response.data)} flight offers")
            flights = []
            for offer in response.data:
                try:
                    for itinerary in offer['itineraries']:
                        segment = itinerary['segments'][0]  # Taking first segment for direct flights
                        duration, duration_min = self._format_duration(segment['duration'])
                        
                        airline_code = segment['carrierCode']
                        airline_name = self._get_airline_name(airline_code)
                        
                        # Use airline code if name not found in our mapping
                        if airline_name == airline_code:
                            airline_name = f"{airline_code} Airlines"
                        
                        flight = {
                            'FlightNumber': f"{airline_code}{segment['number']}",
                            'Origin': origin,
                            'Destination': destination,
                            'DepartureDate': segment['departure']['at'].split('T')[0],
                            'DepartureTime': segment['departure']['at'].split('T')[1],
                            'ArrivalDate': segment['arrival']['at'].split('T')[0],
                            'ArrivalTime': segment['arrival']['at'].split('T')[1],
                            'Price': float(offer['price']['total']),
                            'Airline': airline_name,
                            'Duration': duration,
                            'Duration_min': duration_min
                        }
                        flights.append(flight)
                except KeyError as ke:
                    logging.warning(f"Skipping malformed flight offer: {ke}")
                    continue

            logging.info(f"Returning {len(flights)} valid flights")
            return flights

        except ResponseError as error:
            logging.error(f"Amadeus API error: {error}")
            raise Exception(f"Unable to fetch flights: {error}")
        except Exception as e:
            logging.error(f"Error fetching flights: {e}")
            raise Exception(f"An error occurred while searching for flights: {e}")
