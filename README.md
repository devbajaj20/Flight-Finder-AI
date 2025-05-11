# Flight Finder AI 🛫

An intelligent flight search assistant powered by AI that helps users find flights using natural language queries. The application combines the power of Google's Gemini AI with real-time flight data from Amadeus to provide a seamless flight search experience.

## ✨ Features

### Natural Language Search
- Search flights using everyday language
- Support for complex queries like "morning flights from Delhi to Mumbai under ₹5000"
- AI-powered understanding of user preferences

### Smart Filtering
- **Price Filtering**: Find flights within your budget
  - Support for phrases like "under", "less than", "cheaper than", "below"
  - Automatic currency formatting with Indian Rupees (₹)

- **Time-based Filtering**:
  - Morning flights (before 12:00)
  - Afternoon flights (12:00 - 17:00)
  - Night flights (after 17:00)
  - Natural language support ("morning", "afternoon", "evening", "night")

### Flight Information
- Comprehensive flight details including:
  - Flight number
  - Origin and destination
  - Departure and arrival times
  - Duration
  - Price
  - Airline

### Supported Cities
- Delhi
- Mumbai
- Bangalore
- Chennai
- Kolkata
- Hyderabad
- Pune
- Jaipur

### Supported Airlines
- Air India
- SpiceJet
- IndiGo
- Vistara
- GoAir
- AirAsia

## 🚀 Getting Started

### Prerequisites
- Python 3.12 or higher
- Google Gemini API key
- Amadeus API key

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yashs79/Ai-flight-finder.git
   cd Ai-flight-finder
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   AMADEUS_API_KEY=your_amadeus_api_key
   ```

5. Start the FastAPI backend:
   ```bash
   uvicorn app:app --reload --port 8000
   ```

6. Start the Chainlit interface (in a new terminal):
   ```bash
   chainlit run app.py --port 8001
   ```

7. Access the application:
   - FastAPI backend: http://localhost:8000
   - Chainlit interface: http://localhost:8001

## 🐳 Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t flight-finder .
   ```

2. Run the container:
   ```bash
   docker run -p 8001:8001 \
     -e GOOGLE_API_KEY=your_gemini_api_key \
     -e AMADEUS_API_KEY=your_amadeus_api_key \
     flight-finder
   ```

## 🌐 Railway.app Deployment

1. Install Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```

2. Login and initialize:
   ```bash
   railway login
   railway init
   ```

3. Set environment variables in Railway dashboard:
   - `GOOGLE_API_KEY`
   - `AMADEUS_API_KEY`

4. Deploy:
   ```bash
   railway up
   ```

## 🛠️ Tech Stack

- **Backend**:
  - Python 3.12
  - FastAPI - High-performance web framework
  - Chainlit - Interactive chat interface
  - Uvicorn - ASGI server

- **AI & APIs**:
  - Google Gemini API - Natural language processing
  - Amadeus API - Real-time flight data

- **Development Tools**:
  - Docker - Containerization
  - Railway.app - Cloud deployment

## 📝 Example Queries

- "Show me morning flights from Delhi to Mumbai"
- "Find flights from Bangalore to Chennai under ₹5000"
- "Are there any evening flights from Kolkata to Hyderabad tomorrow?"
- "Show me the cheapest flights from Pune to Jaipur"
- "What are the available Air India flights from Delhi?"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## API Keys Required

To run this application, you'll need to obtain API keys from the following services:

1. **Google API Key (Gemini)**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file as `GOOGLE_API_KEY`

2. **Amadeus API Key**
   - Visit [Amadeus for Developers](https://developers.amadeus.com/)
   - Sign up for a free account
   - Create a new API key
   - Add it to your `.env` file as `AMADEUS_API_KEY`

3. **Firebase API Key** (if using Firebase features)
   - Visit [Firebase Console](https://console.firebase.google.com/)
   - Create a new project
   - Get your API key from project settings
   - Add it to your `.env` file as `FIREBASE_API_KEY`

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
GOOGLE_API_KEY=your_gemini_api_key
AMADEUS_API_KEY=your_amadeus_api_key
FIREBASE_API_KEY=your_firebase_key_here
```

⚠️ **Important**: Never commit your actual API keys to version control. The `.env` file is already included in `.gitignore` to prevent accidental commits.
