                 # For loading and saving machine learning models
import yfinance as yf         # Yahoo Finance API wrapper for stock data
from flask import Flask, render_template, request, jsonify, redirect, url_for, session  # Flask web framework components
from datetime import datetime, timedelta, timezone  # For date and time operations
import pandas as pd           # For data manipulation and analysis
import hashlib               # For password encryption
from flask_sqlalchemy import SQLAlchemy  # SQL database ORM for Flask
import time                  # For time-related operations
import requests              # For making HTTP requests
from requests.adapters import HTTPAdapter  # For configuring HTTP requests
from requests.packages.urllib3.util.retry import Retry  # For implementing retry logic
import threading  
import random                # For generating random numbers
import numpy as np           # For numerical operations
import cachetools            # For TTL cache implementation

# Create and configure Flask application
# Create and configure Flask application
app = Flask(__name__)        # Initialize Flask app
app.secret_key = 'your_secret_key'  # Set secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # Configure SQLite database location
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable SQLAlchemy modification tracking

db = SQLAlchemy(app)

# Define rate limiting variables and caching
rate_limit_lock = threading.Lock()  # Create thread lock for synchronization
request_lock = threading.Lock()     # Lock for in-progress requests
last_request_time = {}        # Dictionary to store last request timestamps
in_progress_requests = {}     # Track requests in progress
MIN_REQUEST_INTERVAL = 1      # Minimum time (in seconds) between requests
CACHE_TTL = 60                # Cache data for 60 seconds
stock_data_cache = cachetools.TTLCache(maxsize=100, ttl=CACHE_TTL)

# Define User database model
class User(db.Model):
    """User model for storing user data"""
    id = db.Column(db.Integer, primary_key=True)  # Primary key for user
    username = db.Column(db.String(150), unique=True, nullable=False)  # Unique username
    password = db.Column(db.String(150), nullable=False)  # Hashed password
    preferred_stocks = db.Column(db.PickleType, nullable=True)  # List of user's preferred stocks

# Define Prediction database model
# Define Prediction database model
class Prediction(db.Model):
    """Model for storing prediction data"""
    id = db.Column(db.Integer, primary_key=True)  # Primary key for prediction
    symbol = db.Column(db.String(50), nullable=False)  # Stock symbol
    buy = db.Column(db.Boolean, nullable=False)  # Buy (True) or Sell (False) signal
    future_predictions = db.Column(db.PickleType, nullable=False)  # List of future price predictions
    # Store timestamp in UTC
    timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))  # When prediction was made

# Define StockData database model
class StockData(db.Model):
    """Model for caching stock data"""
    id = db.Column(db.Integer, primary_key=True)  # Primary key for stock data
    symbol = db.Column(db.String(50), nullable=False)  # Stock symbol
    data = db.Column(db.PickleType, nullable=False)  # Cached stock data
    # Store timestamp in UTC
    timestamp = db.Column(db.DateTime, default=datetime.now(timezone.utc))  # When data was cached

# Create all database tables
with app.app_context():
    db.create_all()

# List of stock symbols to monitor
symbols = ['TATASTEEL.NS', 'HDFCBANK.NS', 'WIPRO.NS', 'TCS.NS', 'INFY.NS']

# Dictionaries to store models and scalers
models = {}      # Store ML models for each stock
scalers = {}     # Store data scalers for each stock

# Load ML models and scalers for each stock symbol
for symbol in symbols:
    try:
        # Update the paths to use the XGBoost models saved from train.py
        models[symbol] = joblib.load(f'xgb_{symbol}_classifier.pkl')
        scalers[symbol] = joblib.load(f'scaler_{symbol}.pkl')
        print(f"Loaded XGBoost model and scaler for {symbol}.")
    except FileNotFoundError:
        print(f"Model or scaler for {symbol} not found. Ensure the files are in the correct location.")

# Helper functions
def hash_password(password):
    """Hash password using SHA-256 encryption"""
    return hashlib.sha256(password.encode()).hexdigest()

def fetch_live_data(symbol):
    """Fetch real-time stock data with improved handling"""
    global last_request_time, stock_data_cache, in_progress_requests
    
    # Check cache first
    cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    with request_lock:
        # Return cached data if available and not too old
        if cache_key in stock_data_cache:
            cached_data = stock_data_cache[cache_key]
            cache_time = datetime.strptime(cached_data[0]['Datetime'], '%Y-%m-%d %H:%M:%S') if isinstance(cached_data[0].get('Datetime'), str) else cached_data[0].get('Datetime')
            if isinstance(cache_time, datetime) and datetime.now() - cache_time < timedelta(minutes=1):
                print(f"Returning cached data for {symbol}")
                return cached_data
        
        # Wait if there's already a request in progress for this symbol
        if symbol in in_progress_requests:
            print(f"Request already in progress for {symbol}, waiting...")
            wait_start = time.time()
            while symbol in in_progress_requests:
                time.sleep(0.1)
                if time.time() - wait_start > 1:  # Timeout after 10 seconds
                    print(f"Timeout waiting for in-progress request for {symbol}")
                    break
            if cache_key in stock_data_cache:
                return stock_data_cache[cache_key]
        
        # Mark this request as in progress
        in_progress_requests[symbol] = True
    
    try:
        # Implement rate limiting
        with rate_limit_lock:  # Thread-safe operation
            current_time = time.time()  # Get current timestamp
            if symbol in last_request_time:  # Check if symbol has been requested before
                time_since_last = current_time - last_request_time[symbol]  # Calculate time since last request
                if time_since_last < MIN_REQUEST_INTERVAL:  # If too soon for new request
                    sleep_time = MIN_REQUEST_INTERVAL - time_since_last  # Calculate wait time
                    time.sleep(sleep_time)  # Wait before making new request
            last_request_time[symbol] = time.time()  # Update last request time

        # Try yfinance ticker method first
        try:
            print(f"Attempting yfinance ticker method for {symbol}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1d', interval='1m', prepost=True, timeout=5)
            
            if not df.empty:
                # Process the DataFrame into our standard format
                df.reset_index(inplace=True)
                records = []
                for _, row in df.iterrows():
                    try:
                        record = {
                            'Datetime': pd.to_datetime(row['Datetime']).strftime('%Y-%m-%d %H:%M:%S'),
                            'Open': float(row['Open']),
                            'High': float(row['High']),
                            'Low': float(row['Low']),
                            'Close': float(row['Close']),
                            'Volume': float(row['Volume'])
                        }
                        records.append(record)
                    except (KeyError, ValueError, TypeError) as e:
                        print(f"Error processing row: {e}")
                        continue
                
                if records:
                    print(f"Successfully fetched {len(records)} records from yfinance ticker for {symbol}")
                    stock_data_cache[cache_key] = records
                    return records
        except Exception as e:
            print(f"yfinance ticker error: {str(e)}")
        
        # If yfinance ticker fails, try yfinance download
        try:
            print(f"Attempting yfinance download for {symbol}")
            end = datetime.now()
            start = end - timedelta(hours=1)
            
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval='1m',
                progress=False,
                timeout=1,
                prepost=True,
                threads=False
            )
            
            if not df.empty:
                df.reset_index(inplace=True)
                records = []
                for _, row in df.iterrows():
                    try:
                        record = {
                            'Datetime': pd.to_datetime(row['Datetime']).strftime('%Y-%m-%d %H:%M:%S'),
                            'Open': float(row['Open']),
                            'High': float(row['High']),
                            'Low': float(row['Low']),
                            'Close': float(row['Close']),
                            'Volume': float(row['Volume'])
                        }
                        records.append(record)
                    except (KeyError, ValueError, TypeError) as e:
                        print(f"Error processing row: {e}")
                        continue
                
                if records:
                    print(f"Successfully fetched {len(records)} records from yfinance download for {symbol}")
                    stock_data_cache[cache_key] = records
                    return records
        except Exception as e:
            print(f"yfinance download error: {str(e)}")
        
        # If yfinance methods fail, try Yahoo Finance API
        try:
            print(f"Attempting Yahoo Finance API for {symbol}")
            session = requests.Session()
            retries = Retry(
                total=3,
                backoff_factor=1.5,
                status_forcelist=[429, 500, 502, 503, 504],
                respect_retry_after_header=True
            )
            session.mount('http://', HTTPAdapter(max_retries=retries))
            session.mount('https://', HTTPAdapter(max_retries=retries))
            
            url = f'https://query2.finance.yahoo.com/v8/finance/chart/{symbol}'
            params = {
                'range': '1d',
                'interval': '1m',
                'includePrePost': 'true'
            }
            
            # Use randomized User-Agent to avoid rate limiting
            headers = {
                'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 100)}.0.{random.randint(4000, 5000)}.{random.randint(100, 200)} Safari/537.36'
            }
            
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                timestamps = result.get('timestamp', [])
                quotes = result.get('indicators', {}).get('quote', [{}])[0]
                
                if timestamps and all(key in quotes for key in ['open', 'high', 'low', 'close', 'volume']):
                    records = []
                    for i in range(len(timestamps)):
                        if all(quotes[key][i] is not None for key in ['open', 'high', 'low', 'close', 'volume']):
                            record = {
                                'Datetime': datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M:%S'),
                                'Open': float(quotes['open'][i]),
                                'High': float(quotes['high'][i]),
                                'Low': float(quotes['low'][i]),
                                'Close': float(quotes['close'][i]),
                                'Volume': float(quotes['volume'][i])
                            }
                            records.append(record)
                    
                    if records:
                        print(f"Successfully fetched {len(records)} records from Yahoo Finance API for {symbol}")
                        stock_data_cache[cache_key] = records
                        return records
        except Exception as e:
            print(f"Yahoo Finance API error: {str(e)}")
        
        # If all methods fail, generate dummy data
        print(f"All data fetching methods failed for {symbol}, using dummy data")
        dummy_data = generate_dummy_data(symbol)
        stock_data_cache[cache_key] = dummy_data
        return dummy_data
        
    finally:
        # Always remove the in-progress flag
        with request_lock:
            in_progress_requests.pop(symbol, None)

def generate_dummy_data(symbol):
    """Generate realistic dummy data based on symbol"""
    print(f"Generating dummy data for {symbol}")
    
    # Base prices for different symbols
    base_prices = {
        'TATASTEEL.NS': 800.0,
        'HDFCBANK.NS': 1500.0,
        'WIPRO.NS': 450.0,
        'TCS.NS': 3500.0,
        'INFY.NS': 1300.0
    }
    
    base_price = base_prices.get(symbol, 1000.0)
    current_time = datetime.now()
    
    dummy_data = []
    last_price = base_price
    
    for i in range(60):
        timestamp = current_time - timedelta(minutes=i)
        # More realistic price movements
        price_change = np.random.normal(0, 0.0015)
        price = last_price * (1 + price_change)
        last_price = price
        
        dummy_data.append({
            'Datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Open': round(price * (1 + np.random.normal(0, 0.0002)), 2),
            'High': round(price * (1 + abs(np.random.normal(0, 0.0005))), 2),
            'Low': round(price * (1 - abs(np.random.normal(0, 0.0005))), 2),
            'Close': round(price, 2),
            'Volume': int(np.random.normal(100000, 20000))
        })
    
    return dummy_data[::-1]  # Return in chronological order

# Route handlers
@app.route('/')
def index():
    """Homepage route handler"""
    if 'username' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['username']).first()
    preferred_stocks = user.preferred_stocks if user else []
    return render_template('dashboard.html', stocks=symbols, preferred_stocks=preferred_stocks)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        preferred_stocks = request.form.getlist('preferred_stocks')
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            print(f"Registration failed: Username {username} already exists.")
            return 'Username already exists'
        hashed_password = hash_password(password)
        new_user = User(username=username, password=hashed_password, preferred_stocks=preferred_stocks)
        db.session.add(new_user)
        db.session.commit()
        print(f"User {username} registered successfully.")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['username'] = username
            print(f"User {username} logged in successfully.")
            return redirect(url_for('index'))
        print(f"Login failed: Invalid credentials for {username}.")
        return 'Invalid credentials'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    print("User logged out.")
    return redirect(url_for('login'))

@app.route('/stock-data/<symbol>')
def stock_data(symbol):
    data = fetch_live_data(symbol)
    if isinstance(data, dict) and 'error' in data:
        return jsonify(data)
    
    # Format is already standardized from fetch_live_data
    return jsonify(data)

@app.route('/predict/<symbol>')
def predict(symbol):
    if symbol not in models or symbol not in scalers:
        print(f"Model or scaler for {symbol} not loaded.")
        return jsonify({"error": f"Model or scaler for {symbol} not loaded."})

    try:
        print(f"Fetching live data for {symbol}.")
        data = fetch_live_data(symbol)
        if isinstance(data, dict) and 'error' in data:
            print(f"Error fetching live data for {symbol}: {data['error']}")
            return jsonify(data)

        # Get latest data
        latest_data = data[-1]
        
        # Get previous data for percent change calculation
        prev_data = data[-2] if len(data) > 1 else latest_data
        
        # Extract values
        prev_close = float(latest_data['Close'])
        volume = float(latest_data['Volume'])
        percent_change = ((latest_data['Close'] - prev_data['Close']) / prev_data['Close']) * 100 if prev_data != latest_data else 0

        # Prepare input features
        input_features = [[prev_close, volume, percent_change]]
        
        # Scale features using the saved scaler
        input_features_scaled = scalers[symbol].transform(input_features)
        
        # Get prediction probabilities from XGBoost model
        probabilities = models[symbol].predict_proba(input_features_scaled)[0]
        
        # Calculate confidence based on probability difference
        buy_prob = probabilities[1]
        sell_prob = probabilities[0]
        
        # Enhanced confidence calculation
        if buy_prob > sell_prob:
            confidence = round(buy_prob * 100, 2)
            signal = 'buy'
        else:
            confidence = round(sell_prob * 100, 2)
            signal = 'sell'
            
        # Calculate trend
        if len(data) > 1:
            closes = [record['Close'] for record in data]
            trend = ((closes[-1] - closes[0]) / closes[0]) * 100
        else:
            trend = 0
            
        # Generate future predictions
        future_predictions = []
        future_date = datetime.now(timezone.utc)
        current_close = latest_data['Close']
        
        for _ in range(5):
            future_date += timedelta(days=1)
            # Add some randomness for more realistic predictions
            predicted_change = (trend / 100) * current_close * (1 + np.random.normal(0, 0.02))
            current_close += predicted_change
            
            future_predictions.append({
                'date': future_date.isoformat(),
                'predicted_close': round(current_close, 2)
            })

        return jsonify({
            'signal': signal,
            'confidence': confidence,
            'future_predictions': future_predictions,
            'current_price': round(latest_data['Close'], 2)
        })

    except Exception as e:
        print(f"Error during prediction for {symbol}: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/stocks')
def api_stocks():
    stock_data = {}
    for symbol in symbols:
        print(f"Fetching data for {symbol}.")
        data = fetch_live_data(symbol)
        if isinstance(data, dict) and 'error' in data:
            print(f"Error fetching data for {symbol}: {data['error']}")
            stock_data[symbol] = {'error': data['error']}
        else:
            try:
                # Get latest data and previous data
                latest_data = data[-1]
                prev_data = data[-2] if len(data) > 1 else latest_data
                
                # Extract required values
                current_close = float(latest_data['Close'])
                prev_close = float(prev_data['Close']) if prev_data != latest_data else current_close
                volume = float(latest_data['Volume'])
                percent_change = ((current_close - prev_close) / prev_close * 100) if prev_close > 0 else 0

                # Prepare input features for prediction
                input_features = [[prev_close, volume, percent_change]]
                
                # Scale features using the saved scaler
                input_features_scaled = scalers[symbol].transform(input_features)
                
                # Get prediction probabilities
                probabilities = models[symbol].predict_proba(input_features_scaled)[0]
                
                # Determine confidence and signal
                buy_prob = probabilities[1]
                sell_prob = probabilities[0]
                confidence = round(max(buy_prob, sell_prob) * 100, 2)
                
                if buy_prob > sell_prob:
                    signal = 'Buy'
                    signal_color = 'green'
                elif confidence > 60:
                    signal = 'Sell'
                    signal_color = 'red'
                else:
                    signal = 'Hold'
                    signal_color = 'yellow'

                # Generate future predictions
                future_predictions = []
                future_date = datetime.now(timezone.utc)
                price_prediction = current_close
                
                # Calculate trend from recent data
                if len(data) > 1:
                    closes = [record['Close'] for record in data]
                    trend = ((closes[-1] - closes[0]) / closes[0]) * 100 / len(closes)  # Daily percent change
                else:
                    trend = 0.5  # Default small uptrend
                
                for i in range(5):
                    future_date += timedelta(days=1)
                    # Add some randomness to predictions
                    rand_factor = 1 + np.random.normal(0, 0.02)  # Normal distribution with 2% std dev
                    day_change = (trend / 100) * price_prediction * rand_factor
                    price_prediction += day_change
                    
                    future_predictions.append({
                        'date': future_date.strftime('%Y-%m-%d'),
                        'predicted_close': round(price_prediction, 2)
                    })

                stock_data[symbol] = {
                    'signal': signal,
                    'signal_color': signal_color,
                    'confidence': confidence,
                    'current_price': round(current_close, 2),
                    'percent_change': round(percent_change, 2),
                    'volume': int(volume),
                    'future_predictions': future_predictions
                }
                
            except Exception as e:
                print(f"Error processing data for {symbol}: {e}")
                stock_data[symbol] = {'error': str(e)}

    print(f"API response ready with data for {len(stock_data)} symbols")
    return jsonify(stock_data)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
