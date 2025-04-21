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
import plotly.graph_objects as go
import networkx as nx

# Constants for grid pricing
GRID_BUY_PRICE = 0.10  # Grid buys surplus from producers at $0.10 per kWh
GRID_SELL_PRICE = 0.20  # Grid sells energy to consumers at $0.20 per kWh
SELLER_PRICE_LOW = GRID_BUY_PRICE + 0.01  # 0.11
SELLER_PRICE_HIGH = GRID_SELL_PRICE - 0.01  # 0.19
BUYER_PRICE_LOW = SELLER_PRICE_LOW  # 0.11
BUYER_PRICE_HIGH = GRID_SELL_PRICE  # 0.20

def hash_household_id(household_id):
    return hashlib.sha256(str(household_id).encode()).hexdigest()

def process_energy_data(insert_to_db=True):
    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    energy_collection = db["energydata"]
    raw_documents = list(energy_collection.find())
    df = pd.DataFrame(raw_documents)
    required_cols = ["solarPower", "windPower", "powerConsumption", "voltage", "current",
                     "electricityPrice", "overloadCondition", "transformerFault", "householdId"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    if not datetime_cols.empty:
        df[datetime_cols] = df[datetime_cols].fillna(pd.NaT)
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
    fields = ["householdId", "householdId_hash", "NetPower", "Efficiency", "OverloadRisk", "AdjCost",
              "NoFault", "BothFaults", "OverloadOnly", "TransformerFaultOnly", "Price", "Role"]
    if insert_to_db:
        results_collection = db["energy_trading"]
        records = df[fields].to_dict(orient="records")
        results_collection.delete_many({})
        results_collection.insert_many(records)
    return df[fields]

def merge(left, right, compare):
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
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], compare)
    right = merge_sort(arr[mid:], compare)
    return merge(left, right, compare)

def prepare_households_for_trading(df):
    households = {}
    for _, row in df.iterrows():
        household_id = str(row['householdId'])
        net_power = float(row['NetPower'])
        role = 'seller' if net_power > 0 else 'buyer'
        households[f"H{household_id}"] = {
            'householdId': household_id,
            'role': role,
            'net': net_power,
            'remaining': net_power,
            'price': float(row['Price']),
            'traded_units': 0,
            'p2p_traded_units': 0,
            'total_price': 0.0,
            'no_fault': bool(row['NoFault']),
            'both_faults': bool(row['BothFaults'])
        }
    return households

def perform_trading(households):
    eligible_households = {name: data for name, data in households.items() if data['no_fault']}
    sellers = {n: d for n, d in eligible_households.items() if d['role'] == 'seller' and d['net'] > 0}
    buyers = {n: d for n, d in eligible_households.items() if d['role'] == 'buyer' and d['net'] < 0}

    # Sort sellers by net energy descending (highest surplus first)
    compare_seller = lambda a, b: a[1]['net'] > b[1]['net']
    sorted_sellers = merge_sort(list(sellers.items()), compare_seller)
    
    # Sort buyers by net energy ascending (most negative first)
    compare_buyer = lambda a, b: a[1]['net'] < b[1]['net']
    sorted_buyers = merge_sort(list(buyers.items()), compare_buyer)

    trades = []
    # P2P Trading
    for s_name, s_data in sorted_sellers:
        for b_name, b_data in sorted_buyers:
            if s_data['remaining'] > 0 and b_data['remaining'] < 0:
                if b_data['price'] >= s_data['price']:
                    trade_qty = min(s_data['remaining'], -b_data['remaining'])
                    traded_price = np.round((s_data['price'] + b_data['price']) / 2, 2)
                    s_data['traded_units'] += trade_qty
                    s_data['p2p_traded_units'] += trade_qty
                    s_data['total_price'] += trade_qty * traded_price
                    s_data['remaining'] -= trade_qty
                    b_data['traded_units'] += trade_qty
                    b_data['p2p_traded_units'] += trade_qty
                    b_data['total_price'] -= trade_qty * traded_price
                    b_data['remaining'] += trade_qty
                    trades.append({
                        'seller': s_name,
                        'buyer': b_name,
                        'quantity': trade_qty,
                        'price': traded_price,
                        'type': 'p2p'
                    })

    # Grid Trading: Remaining surplus
    for s_name, s_data in sellers.items():
        if s_data['remaining'] > 0:
            qty = s_data['remaining']
            s_data['traded_units'] += qty
            s_data['total_price'] += qty * GRID_BUY_PRICE
            trades.append({
                'seller': s_name,
                'buyer': 'grid',
                'quantity': qty,
                'price': GRID_BUY_PRICE,
                'type': 'grid'
            })
            s_data['remaining'] = 0

    # Grid Trading: Remaining demand
    for b_name, b_data in buyers.items():
        if b_data['remaining'] < 0:
            qty = -b_data['remaining']
            b_data['traded_units'] += qty
            b_data['total_price'] -= qty * GRID_SELL_PRICE
            trades.append({
                'seller': 'grid',
                'buyer': b_name,
                'quantity': qty,
                'price': GRID_SELL_PRICE,
                'type': 'grid'
            })
            b_data['remaining'] = 0

    return eligible_households, trades

def analyze_market_equilibrium(trades):
    p2p_vol = sum(t['quantity'] for t in trades if t['type'] == 'p2p')
    grid_vol = sum(t['quantity'] for t in trades if t['type'] == 'grid')
    total_vol = p2p_vol + grid_vol
    return {
        'total_trades': len(trades),
        'p2p_trades': sum(1 for t in trades if t['type'] == 'p2p'),
        'grid_trades': sum(1 for t in trades if t['type'] == 'grid'),
        'p2p_volume': round(p2p_vol, 2),
        'grid_volume': round(grid_vol, 2),
        'total_volume': round(total_vol, 2),
        'p2p_percentage': round((p2p_vol / total_vol * 100) if total_vol else 0, 1),
        'grid_percentage': round((grid_vol / total_vol * 100) if total_vol else 0, 1)
    }

def generate_network_graph_html(trades, households):
    G = nx.DiGraph()
    for name, data in households.items():
        G.add_node(name, role=data['role'])
    G.add_node('grid', role='grid')
    for trade in trades:
        G.add_edge(trade['seller'], trade['buyer'],
                   quantity=trade['quantity'],
                   price=trade['price'],
                   type=trade['type'])
    pos = nx.spring_layout(G, seed=42)
    if 'grid' in pos:
        pos['grid'] = np.array([0, 0])
    edge_x_p2p, edge_y_p2p, edge_text_p2p = [], [], []
    edge_x_grid, edge_y_grid, edge_text_grid = [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if edge[2]['type'] == 'p2p':
            edge_x_p2p += [x0, x1, None]
            edge_y_p2p += [y0, y1, None]
            edge_text_p2p.append(f"{edge[0]}→{edge[1]}: {edge[2]['quantity']} kWh @ ${edge[2]['price']:.2f}")
        else:
            edge_x_grid += [x0, x1, None]
            edge_y_grid += [y0, y1, None]
            edge_text_grid.append(f"{edge[0]}→{edge[1]}: {edge[2]['quantity']} kWh @ ${edge[2]['price']:.2f}")
    edge_trace_p2p = go.Scatter(
        x=edge_x_p2p, y=edge_y_p2p,
        line=dict(width=1.5, color='purple'),
        hoverinfo='text',
        text=edge_text_p2p,
        mode='lines',
        name='P2P Trades'
    )
    edge_trace_grid = go.Scatter(
        x=edge_x_grid, y=edge_y_grid,
        line=dict(width=1.5, color='gray', dash='dash'),
        hoverinfo='text',
        text=edge_text_grid,
        mode='lines',
        name='Grid Trades'
    )
    node_x_seller, node_y_seller, node_text_seller = [], [], []
    node_x_buyer, node_y_buyer, node_text_buyer = [], [], []
    node_x_grid, node_y_grid, node_text_grid = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        role = G.nodes[node]['role']
        if node == 'grid':
            node_x_grid.append(x)
            node_y_grid.append(y)
            node_text_grid.append("Grid")
        elif role == 'seller':
            node_x_seller.append(x)
            node_y_seller.append(y)
            node_text_seller.append(f"{node}: Seller")
        elif role == 'buyer':
            node_x_buyer.append(x)
            node_y_buyer.append(y)
            node_text_buyer.append(f"{node}: Buyer")
    seller_trace = go.Scatter(
        x=node_x_seller, y=node_y_seller,
        mode='markers+text',
        text=node_text_seller,
        textposition="top center",
        marker=dict(size=20, color='green'),
        name='Sellers (Producers)'
    )
    buyer_trace = go.Scatter(
        x=node_x_buyer, y=node_y_buyer,
        mode='markers+text',
        text=node_text_buyer,
        textposition="top center",
        marker=dict(size=20, color='red'),
        name='Buyers (Consumers)'
    )
    grid_trace = go.Scatter(
        x=node_x_grid, y=node_y_grid,
        mode='markers+text',
        text=node_text_grid,
        textposition="top center",
        marker=dict(size=25, color='blue'),
        name='Grid'
    )
    fig = go.Figure(
        data=[edge_trace_p2p, edge_trace_grid, seller_trace, buyer_trace, grid_trace],
        layout=go.Layout(
            title="Energy Trading Network",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(x=0, y=1.1, orientation="h"),
            margin=dict(l=20, r=20, t=50, b=20)
        )
    )
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            perp_dx = -dy / length
            perp_dy = dx / length
        else:
            perp_dx, perp_dy = 0, 0
        offset = 0.05
        label_x = mid_x + offset * perp_dx
        label_y = mid_y + offset * perp_dy
        quantity = edge[2]['quantity']
        fig.add_annotation(
            x=label_x, y=label_y,
            text=f"{quantity} kWh",
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor='rgba(0, 0, 0, 0.5)',
            borderwidth=1,
            borderpad=2,
            align='center'
        )
    graph_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return graph_html

def energy_report(request):
    try:
        df_processed = process_energy_data(insert_to_db=True)
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB_NAME]
        results_collection = db["energy_trading"]
        data_to_display = list(results_collection.find({}, {"_id": 0, "householdId_hash": 0}))
        paginator = Paginator(data_to_display, 5)
        page_number = request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)
        game_results = request.session.get('trading_game_results', None)
        if 'trading_game_results' in request.session:
            del request.session['trading_game_results']
        context = {
            "data": data_to_display,
            "page_obj": page_obj,
            "paginator": paginator,
            "game_results": game_results
        }
        return render(request, "report.html", context)
    except Exception as e:
        return render(request, "report.html", {"error": f"An error occurred: {e}"})

def update_energy_trading_collection(request):
    try:
        df_processed = process_energy_data(insert_to_db=True)
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB_NAME]
        results_collection = db["energy_trading"]
        count = results_collection.count_documents({})
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
    try:
        household_id = request.POST.get('household_id')
        if not household_id:
            return JsonResponse({"status": "error", "message": "No household ID provided"}, status=400)
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.MONGO_DB_NAME]
        results_collection = db["energy_trading"]
        selected_household = results_collection.find_one({"householdId": household_id})
        if not selected_household:
            return JsonResponse({"status": "error", "message": "Household not found"}, status=404)
        if selected_household.get('NoFault', 0) != 1:
            messages.warning(request, f"Household {household_id} has faults and cannot participate in trading.")
            return redirect(reverse('energy_report'))
        request.session['selected_household_id'] = household_id
        messages.success(request, f"Household {household_id} selected successfully.")
        return redirect(reverse('energy_report'))
    except Exception as e:
        messages.error(request, f"Error selecting household: {str(e)}")
        return redirect(reverse('energy_report'))

def run_trading_simulation(request):
    try:
        df_processed = process_energy_data(insert_to_db=False)
        households = prepare_households_for_trading(df_processed)
        traded_households, trades = perform_trading(households)
        market_analysis = analyze_market_equilibrium(trades)
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
        return JsonResponse({
            "status": "success",
            "households": households_data,
            "trades": trades,
            "market_analysis": market_analysis
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
@require_POST
def start_trading_game(request):
    try:
        household_ids = []
        for key, value in request.POST.items():
            if key.startswith('household_ids['):
                household_ids.append(value)
        if not household_ids:
            return JsonResponse({"success": False, "error": "No households selected"}, status=400)
        df_processed = process_energy_data(insert_to_db=False)
        selected_df = df_processed[df_processed['householdId'].astype(str).isin(household_ids)]
        if selected_df.empty:
            return JsonResponse({"success": False, "error": "Selected households not found in dataset"}, status=404)
        fault_households = selected_df[selected_df['NoFault'] != 1]['householdId'].tolist()
        if fault_households:
            fault_ids = ", ".join(map(str, fault_households))
            messages.warning(request, f"Households with IDs {fault_ids} have faults and will be excluded from trading.")
        households = prepare_households_for_trading(selected_df)
        traded_households, trades = perform_trading(households)
        market_analysis = analyze_market_equilibrium(trades)
        graph_html = generate_network_graph_html(trades, traded_households)
        households_data = []
        for name, data in households.items():
            household_info = {
                'household_id': data['householdId'],
                'role': data['role'],
                'original_energy': data['net'],
                'price': data['price'],
                'status': 'excluded due to faults' if not data['no_fault'] else 'traded',
            }
            if data['no_fault'] and name in traded_households:
                traded_data = traded_households[name]
                household_info.update({
                    'total_traded': traded_data['traded_units'],
                    'p2p_traded': traded_data['p2p_traded_units'],
                    'grid_traded': traded_data['traded_units'] - traded_data['p2p_traded_units'],
                    'financial_result': round(traded_data['total_price'], 2),
                })
            else:
                household_info.update({
                    'total_traded': 0,
                    'p2p_traded': 0,
                    'grid_traded': 0,
                    'financial_result': 0.0,
                })
            households_data.append(household_info)
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
            else:
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
        game_results = {
            'households': households_data,
            'p2p_trades': p2p_trades,
            'grid_trades': grid_trades,
            'market_analysis': market_analysis,
            'graph_html': graph_html
        }
        request.session['trading_game_results'] = game_results
        return redirect('energy_report')
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)