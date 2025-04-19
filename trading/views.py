from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.urls import reverse
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from pymongo import MongoClient
import pandas as pd
import hashlib
import numpy as np
import json


def hash_household_id(household_id):
    """
    Generate a SHA-256 hash for the given household ID for privacy.
    """
    return hashlib.sha256(str(household_id).encode()).hexdigest()


def process_energy_data(insert_to_db=True):
    """
    Internal helper to fetch raw energy data, compute derived metrics,
    and return a DataFrame ready for insertion into energy_trading.
    
    Parameters:
    insert_to_db (bool): Flag to control whether data should be inserted into MongoDB
                         Set to False for game simulations to avoid modifying the database
    """
    # Connect to MongoDB
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    energy_collection = db["energydata"]

    # Fetch raw data
    raw_documents = list(energy_collection.find())
    df = pd.DataFrame(raw_documents)

    # Ensure required fields exist
    required_cols = [
        "solarPower", "windPower", "powerConsumption", "voltage", "current",
        "electricityPrice", "overloadCondition", "transformerFault", "householdId"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Fill missing numeric values with 0 to avoid dtype conflicts
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Handle datetime columns explicitly, if any
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    if not datetime_cols.empty:
        df[datetime_cols] = df[datetime_cols].fillna(pd.NaT)

    # Compute derived metrics
    df["NetPower"] = (df["solarPower"] + df["windPower"] - df["powerConsumption"]).round(2)
    df["Efficiency"] = (((df["solarPower"] + df["windPower"]) / df["powerConsumption"]) * 100).round(2)
    df["OverloadRisk"] = (df["powerConsumption"] / (df["voltage"] * df["current"])).round(2)
    df["AdjCost"] = (df["powerConsumption"] * df["electricityPrice"]).round(2)
    df["NoFault"] = ((df["overloadCondition"] == 0) & (df["transformerFault"] == 0)).astype(int)
    df["BothFaults"] = ((df["overloadCondition"] == 1) & (df["transformerFault"] == 1)).astype(int)
    df["OverloadOnly"] = ((df["overloadCondition"] == 1) & (df["transformerFault"] == 0)).astype(int)
    df["TransformerFaultOnly"] = ((df["overloadCondition"] == 0) & (df["transformerFault"] == 1)).astype(int)
    df["Role"] = df["NetPower"].apply(lambda x: "Producer" if x > 0 else "Consumer")
    df["householdId_hash"] = df["householdId"].apply(hash_household_id)
    df["Price"] = df["electricityPrice"].round(2)

    # Select fields for insertion
    fields = [
        "householdId", "householdId_hash", "NetPower", "Efficiency",
        "OverloadRisk", "AdjCost", "NoFault", "BothFaults",
        "OverloadOnly", "TransformerFaultOnly", "Price", "Role"
    ]
    
    # If insert_to_db flag is True, insert the processed data into MongoDB
    if insert_to_db:
        results_collection = db["energy_trading"]
        # Clear old data and insert new
        records = df[fields].to_dict(orient="records")
        results_collection.delete_many({})
        results_collection.insert_many(records)
    
    return df[fields]


# --------------------------- Energy Trading Game Functions ---------------------------
# Constants for energy trading
GRID_BUY_PRICE = 0.10   # Price grid buys surplus energy (EUR/kWh)
GRID_SELL_PRICE = 0.20  # Price grid sells energy (EUR/kWh)


def merge(left, right, compare):
    """
    Merge two sorted arrays based on the provided comparison function.
    Used by the merge sort algorithm for sorting households.
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if compare(left[i], right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(arr, compare):
    """
    Implementation of merge sort algorithm using a custom comparison function.
    Used to sort households by their energy production or consumption levels.
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], compare)
    right = merge_sort(arr[mid:], compare)
    return merge(left, right, compare)


def prepare_households_for_trading(df):
    """
    Convert DataFrame rows into a format suitable for the trading algorithm.
    Each household is identified by a key and contains its trading properties.
    """
    households = {}
    
    # Process each row in the DataFrame
    for _, row in df.iterrows():
        household_id = str(row['householdId'])
        net_power = float(row['NetPower'])
        role = 'seller' if net_power > 0 else 'buyer'
        
        # Create household entry with trading attributes
        households[f"H{household_id}"] = {
            'householdId': household_id,
            'role': role,
            'net': abs(net_power),  # Absolute value for easier sorting
            'remaining': abs(net_power),  # Energy left to trade
            'price': float(row['Price']),
            'traded_units': 0,      # Total energy traded
            'p2p_traded_units': 0,  # Energy traded peer-to-peer
            'total_price': 0.0,     # Revenue or cost from trading
            'no_fault': bool(row['NoFault'])  # Only fault-free can trade
        }
    
    return households


def perform_trading(households):
    """
    Execute the energy trading algorithm using provided household data.
    Returns the updated households and a list of all trades that occurred.
    """
    # Filter households with no faults (eligible for trading)
    eligible_households = {name: data for name, data in households.items() if data.get('no_fault', True)}
    
    # Separate sellers and buyers
    sellers = {n: d for n, d in eligible_households.items() if d['role'] == 'seller' and d['net'] > 0}
    buyers = {n: d for n, d in eligible_households.items() if d['role'] == 'buyer' and d['net'] > 0}

    # Define comparison functions for sorting
    compare_seller = lambda a, b: a[1]['net'] > b[1]['net']  # Sort sellers by amount (descending)
    compare_buyer = lambda a, b: a[1]['net'] < b[1]['net']   # Sort buyers by amount (ascending)

    # Sort sellers and buyers
    sorted_sellers = merge_sort(list(sellers.items()), compare_seller)
    sorted_buyers = merge_sort(list(buyers.items()), compare_buyer)

    trades = []
    
    # Step 1: Peer-to-Peer Trading (Double Auction)
    for s_name, s_data in sorted_sellers:
        for b_name, b_data in sorted_buyers:
            # Check if both have energy left to trade
            if s_data['remaining'] > 0 and b_data['remaining'] > 0:
                # Check if buyer's price offer is acceptable to seller
                if b_data['price'] >= s_data['price']:
                    # Determine trade quantity (minimum of what's available)
                    trade_qty = min(s_data['remaining'], b_data['remaining'])
                    # Calculate average price between seller and buyer
                    traded_price = np.round((s_data['price'] + b_data['price']) / 2, 2)
                    
                    # Update seller records
                    s_data['traded_units'] += trade_qty
                    s_data['p2p_traded_units'] += trade_qty
                    s_data['total_price'] += trade_qty * traded_price  # Revenue
                    s_data['remaining'] -= trade_qty  # Reduce energy left
                    
                    # Update buyer records
                    b_data['traded_units'] += trade_qty
                    b_data['p2p_traded_units'] += trade_qty
                    b_data['total_price'] -= trade_qty * traded_price  # Cost
                    b_data['remaining'] -= trade_qty  # Reduce energy needed
                    
                    # Record the trade
                    trades.append({
                        'seller': s_name,
                        'buyer': b_name,
                        'quantity': trade_qty,
                        'price': traded_price,
                        'type': 'p2p'  # Peer-to-peer trade
                    })

    # Step 2: Grid Trading (Fallback for untraded energy)
    # Sellers sell remaining energy to the grid at lower price
    for s_name, s_data in sellers.items():
        if s_data['remaining'] > 0:
            trade_qty = s_data['remaining']
            s_data['traded_units'] += trade_qty
            s_data['total_price'] += trade_qty * GRID_BUY_PRICE
            s_data['remaining'] = 0
            
            trades.append({
                'seller': s_name,
                'buyer': 'grid',
                'quantity': trade_qty,
                'price': GRID_BUY_PRICE,
                'type': 'grid'
            })

    # Buyers purchase remaining energy needs from grid at higher price
    for b_name, b_data in buyers.items():
        if b_data['remaining'] > 0:
            trade_qty = b_data['remaining']
            b_data['traded_units'] += trade_qty
            b_data['total_price'] -= trade_qty * GRID_SELL_PRICE
            b_data['remaining'] = 0
            
            trades.append({
                'seller': 'grid',
                'buyer': b_name,
                'quantity': trade_qty,
                'price': GRID_SELL_PRICE,
                'type': 'grid'
            })

    return eligible_households, trades


def analyze_market_equilibrium(trades):
    """
    Analyze trading results to understand market dynamics.
    Returns a dictionary with key metrics about the trading session.
    """
    # Calculate p2p and grid trading volumes
    p2p_trades = [t for t in trades if t['type'] == 'p2p']
    grid_trades = [t for t in trades if t['type'] == 'grid']
    
    p2p_volume = sum(t['quantity'] for t in p2p_trades)
    grid_volume = sum(t['quantity'] for t in grid_trades)
    total_volume = p2p_volume + grid_volume
    
    # Calculate percentages
    p2p_percentage = 0 if total_volume == 0 else (p2p_volume / total_volume * 100)
    grid_percentage = 0 if total_volume == 0 else (grid_volume / total_volume * 100)
    
    return {
        'total_trades': len(trades),
        'p2p_trades': len(p2p_trades),
        'grid_trades': len(grid_trades),
        'p2p_volume': round(p2p_volume, 2),
        'grid_volume': round(grid_volume, 2),
        'total_volume': round(total_volume, 2),
        'p2p_percentage': round(p2p_percentage, 1),
        'grid_percentage': round(grid_percentage, 1)
    }


def energy_report(request):
    """
    View to display processed energy trading data in a template.
    Also performs the energy trading simulation and passes results to template.
    """
    try:
        # Process energy data with database insertion
        df_processed = process_energy_data(insert_to_db=True)

        # Connect to MongoDB to fetch data for display
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB_NAME]
        results_collection = db["energy_trading"]

        # Fetch for display, excluding hashed IDs
        data_to_display = list(
            results_collection.find(
                {},
                {"_id": 0, "householdId_hash": 0}
            )
        )
        
        # Set up pagination
        paginator = Paginator(data_to_display, 5)  # Show 5 rows per page
        page_number = request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)
        
        # Get game results from session if available
        game_results = request.session.get('trading_game_results', None)
        
        # Clear game results from session after retrieving them
        if 'trading_game_results' in request.session:
            del request.session['trading_game_results']
        
        # Create context with all data for template
        context = {
            "data": data_to_display,
            "page_obj": page_obj,
            "paginator": paginator,
            "game_results": game_results  # Pass game results to template
        }
        
        return render(request, "report.html", context)

    except Exception as e:
        # Render with error message
        return render(request, "report.html", {"error": f"An error occurred: {e}"})


def update_energy_trading_collection(request):
    """
    API endpoint to refresh the energy_trading MongoDB collection.
    Also performs a trading simulation and returns basic results.
    """
    try:
        # Process data and insert into database
        df_processed = process_energy_data(insert_to_db=True)

        # Connect to check insertion count
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB_NAME]
        results_collection = db["energy_trading"]
        count = results_collection.count_documents({})
            
        # Run trading simulation for API results
        households = prepare_households_for_trading(df_processed)
        traded_households, trades = perform_trading(households)
        market_analysis = analyze_market_equilibrium(trades)

        return JsonResponse({
            "status": "success", 
            "inserted": count,
            "trading_summary": {
                "total_trades": market_analysis['total_trades'],
                "p2p_volume": market_analysis['p2p_volume'],
                "grid_volume": market_analysis['grid_volume'],
                "p2p_percentage": market_analysis['p2p_percentage']
            }
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@require_POST
def select_household(request):
    """
    Handle the selection of a household from the grid participants table.
    Also runs a trading simulation for the selected household.
    """
    try:
        household_id = request.POST.get('household_id')
        
        if not household_id:
            return JsonResponse({"status": "error", "message": "No household ID provided"}, status=400)
        
        # Connect to MongoDB
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB_NAME]
        
        # Find the selected household
        results_collection = db["energy_trading"]
        selected_household = results_collection.find_one({"householdId": household_id})
        
        if not selected_household:
            return JsonResponse({"status": "error", "message": "Household not found"}, status=404)
        
        # Store the selection in session
        request.session['selected_household_id'] = household_id
        messages.success(request, f"Household {household_id} selected successfully.")
        
        # Redirect back to the report page
        return redirect(reverse('energy_report'))
        
    except Exception as e:
        messages.error(request, f"Error selecting household: {str(e)}")
        return redirect(reverse('energy_report'))


def run_trading_simulation(request):
    """
    API endpoint to execute a new energy trading simulation.
    Returns detailed trading results as JSON for frontend visualization.
    """
    try:
        # Get fresh energy data without inserting into MongoDB
        df_processed = process_energy_data(insert_to_db=False)
        
        # Prepare and run trading simulation
        households = prepare_households_for_trading(df_processed)
        traded_households, trades = perform_trading(households)
        market_analysis = analyze_market_equilibrium(trades)
        
        # Format household data for JSON response
        households_data = {}
        for name, data in traded_households.items():
            households_data[name] = {
                'id': data['householdId'],
                'role': data['role'],
                'energy_amount': data['net'],
                'price': data['price'],
                'traded_units': data['traded_units'],
                'p2p_traded': data['p2p_traded_units'],
                'revenue_cost': data['total_price']
            }
        
        # Return detailed results
        return JsonResponse({
            "status": "success",
            "households": households_data,
            "trades": trades,
            "market_analysis": market_analysis
        })
        
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)


@csrf_exempt
@require_POST
def start_trading_game(request):
    """
    Handles the request to start a trading game with selected households.
    Processes selected household IDs from form data, runs the trading simulation,
    and stores results in session for display in report page.
    """
    try:
        # Get selected household IDs from form data
        household_ids = []
        for key, value in request.POST.items():
            if key.startswith('household_ids['):
                household_ids.append(value)
        
        if not household_ids:
            return JsonResponse({
                "success": False,
                "error": "No households selected"
            }, status=400)
        
        # Get full energy data WITHOUT inserting into MongoDB
        df_processed = process_energy_data(insert_to_db=False)
        
        # Filter dataset to only include selected households
        selected_df = df_processed[df_processed['householdId'].astype(str).isin(household_ids)]
        
        if selected_df.empty:
            return JsonResponse({
                "success": False,
                "error": "Selected households not found in dataset"
            }, status=404)
        
        # Prepare filtered households for trading
        households = prepare_households_for_trading(selected_df)
        
        # Run trading simulation
        traded_households, trades = perform_trading(households)
        market_analysis = analyze_market_equilibrium(trades)
        
        # Format households data for template display
        households_data = []
        for name, data in traded_households.items():
            households_data.append({
                'household_id': data['householdId'],
                'role': data['role'],
                'original_energy': data['net'],
                'price': data['price'],
                'total_traded': data['traded_units'],
                'p2p_traded': data['p2p_traded_units'],
                'grid_traded': data['traded_units'] - data['p2p_traded_units'],
                'financial_result': round(data['total_price'], 2),
            })
        
        # Format trades for template display
        p2p_trades = []
        grid_trades = []
        
        for trade in trades:
            if trade['type'] == 'p2p':
                p2p_trades.append({
                    'seller_id': trade['seller'].replace('H', ''),
                    'buyer_id': trade['buyer'].replace('H', ''),
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'total_value': round(trade['quantity'] * trade['price'], 2)
                })
            else:  # grid trade
                grid_trades.append({
                    'participant_id': (
                        trade['seller'].replace('H', '') if trade['seller'] != 'grid' 
                        else trade['buyer'].replace('H', '')
                    ),
                    'role': 'seller' if trade['seller'] != 'grid' else 'buyer',
                    'quantity': trade['quantity'],
                    'price': trade['price'],
                    'total_value': round(trade['quantity'] * trade['price'], 2)
                })
        
        # Create complete game results object
        game_results = {
            'households': households_data,
            'p2p_trades': p2p_trades,
            'grid_trades': grid_trades,
            'market_analysis': market_analysis
        }
        
        # Store in session for retrieval by the report page
        request.session['trading_game_results'] = game_results
        
        # Redirect to the energy_report view
        return redirect('energy_report')
    
    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=500)

# Note: Ensure that the above functions are properly integrated into your Django views.py file.