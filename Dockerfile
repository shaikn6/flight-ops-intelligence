FROM python:3.11-slim

WORKDIR /app

# System deps for geopy / folium
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-generate flight data and train the model at image build time
RUN python -c "from intelligence.flight_data import generate_flights, save_flights; save_flights(generate_flights())" && \
    python -c "from intelligence.delay_predictor import train; train()" && \
    python -m intelligence.map_generator

EXPOSE 8000 8501
