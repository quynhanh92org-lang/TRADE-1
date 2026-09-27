import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import io

# Cố gắng import thư viện kỹ thuật & tối ưu hóa
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Kiểm định Chiến lược EMA & OBV | Cổ phiếu ACB",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện Fintech đẳng cấp & sang trọng
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .tech-badge {
        background-color: #F1F5F9;
        color: #1E293B;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #CBD5E1;
    }
    .tech-badge-primary {
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #BFDBFE;
    }
    .kpi-card {
        background: #FFFFFF;
        padding: 16px 18px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        border-color: #CBD5E1;
    }
    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .kpi-title {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-icon {
        font-size: 1.1rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        margin-bottom: 4px;
        letter-spacing: -0.5px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        font-weight: 500;
    }
    .signal-box-buy {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 1px solid #10B981;
        border-left: 6px solid #059669;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .signal-box-sell {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 1px solid #EF4444;
        border-left: 6px solid #DC2626;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .signal-box-hold {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #3B82F6;
        border-left: 6px solid #2563EB;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .signal-box-cash {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #CBD5E1;
        border-left: 6px solid #64748B;
        padding: 16px 20px;
        border-radius: 10px;
        margin-bottom: 16px;
    }
    .section-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CÁC HÀM XỬ LÝ DỮ LIỆU & TÍNH TOÁN KỸ THUẬT (CORE FUNCTIONS)
# ==============================================================================

@st.cache_data
def load_data(uploaded_file=None):
    """Đọc và xử lý dữ liệu giá cổ phiếu."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Lỗi khi đọc file tải lên: {e}")
            return None
    else:
        file_candidates = ['ACB.csv', './ACB.csv', '../ACB.csv']
        csv_path = None
        for p in file_candidates:
            if os.path.exists(p):
                csv_path = p
                break
        
        if csv_path is None:
            for root, dirs, files in os.walk('.'):
                if 'ACB.csv' in files:
                    csv_path = os.path.join(root, 'ACB.csv')
                    break

        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            st.error("Không tìm thấy file ACB.csv. Vui lòng tải file lên ở thanh bên.")
            return None

    # Kiểm tra cột Date
    date_col = None
    for col in ['Date', 'date', 'Ngay', 'NGAY']:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        st.error("Không tìm thấy cột Ngày (Date) trong dữ liệu.")
        return None

    df['Date'] = pd.to_datetime(df[date_col])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Chuẩn hóa tên cột
    col_mapping = {}
    for col in df.columns:
        c_lower = col.lower()
        if 'close' in c_lower or 'dongcua' in c_lower:
            col_mapping[col] = 'Close'
        elif 'open' in c_lower or 'mocua' in c_lower:
            col_mapping[col] = 'Open'
        elif 'high' in c_lower or 'caonhat' in c_lower:
            col_mapping[col] = 'High'
        elif 'low' in c_lower or 'thapnhat' in c_lower:
            col_mapping[col] = 'Low'
        elif 'volume' in c_lower or 'khoiluong' in c_lower or 'vol' in c_lower:
            col_mapping[col] = 'Volume'

    df.rename(columns=col_mapping, inplace=True)

    required_cols = ['Close', 'Volume']
    for req in required_cols:
        if req not in df.columns:
            st.error(f"Thiếu cột bắt buộc '{req}' trong dữ liệu.")
            return None

    return df


def calculate_indicators(df, ema_period=20, obv_slope_period=3):
    """
    Tính toán các chỉ báo EMA, OBV và Độ dốc OBV (OBV Slope).
    Đồng bộ chuẩn 100% với notebook nghiên cứu.
    """
    data = df.copy()
    close = data['Close']
    volume = data['Volume']

    # 1. Tính EMA
    if HAS_TA:
        ema = ta.trend.ema_indicator(close, window=int(ema_period))
    else:
        ema = close.ewm(span=int(ema_period), adjust=False).mean()

    # 2. Tính OBV
    if HAS_TA:
        obv = ta.volume.on_balance_volume(close, volume)
    else:
        price_diff = close.diff()
        direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
        direction[0] = 0
        obv = (volume * direction).cumsum()
        obv = pd.Series(obv, index=close.index)

    # 3. Tính độ dốc OBV (OBV Slope) = OBV[t] - OBV[t - obv_slope_period]
    obv_slope = obv.diff(int(obv_slope_period))

    data['EMA'] = ema
    data['OBV'] = obv
    data['OBV_Slope'] = obv_slope

    return data


def generate_all_signals(data, ema_period, obv_slope_period):
    """
    Tạo tín hiệu mua/bán cho cả 3 chiến lược:
    1. EMA Riêng lẻ
    2. OBV Riêng lẻ
    3. EMA + OBV Kết hợp

    QUAN TRỌNG: Tất cả tín hiệu đều được shift(1) để loại bỏ Look-ahead bias.
    """
    close = data['Close']
    ema = data['EMA']
    obv_slope = data['OBV_Slope']

    # 1. EMA Riêng lẻ
    raw_entries_ema = (close > ema)
    raw_exits_ema = (close < ema)
    entries_ema = raw_entries_ema.shift(1, fill_value=False)
    exits_ema = raw_exits_ema.shift(1, fill_value=False)

    # 2. OBV Riêng lẻ
    raw_entries_obv = (obv_slope > 0)
    raw_exits_obv = (obv_slope < 0)
    entries_obv = raw_entries_obv.shift(1, fill_value=False)
    exits_obv = raw_exits_obv.shift(1, fill_value=False)

    # 3. EMA + OBV Kết hợp
    raw_entries_comb = (close > ema) & (obv_slope > 0)
    raw_exits_comb = (close < ema)
    entries_comb = raw_entries_comb.shift(1, fill_value=False)
    exits_comb = raw_exits_comb.shift(1, fill_value=False)

    return {
        'EMA': (entries_ema, exits_ema),
        'OBV': (entries_obv, exits_obv),
        'Combined': (entries_comb, exits_comb)
    }

# ==============================================================================
# BỘ MÁY ĐỊNH LƯỢNG CHUYÊN SÂU (ADVANCED QUANTITATIVE ENGINE)
# ==============================================================================

def calculate_top_drawdowns(equity_series, top_n=5):
    """Xác định và xếp hạng các giai đoạn Drawdown sâu nhất."""
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    
    periods = []
    in_drawdown = False
    peak_date = equity_series.index[0]
    trough_date = equity_series.index[0]
    max_dd = 0.0

    for date, val, dd in zip(equity_series.index, equity_series.values, drawdown.values):
        if dd < 0:
            if not in_drawdown:
                in_drawdown = True
                peak_date = prev_date if 'prev_date' in locals() else date
                trough_date = date
                max_dd = dd
            else:
                if dd < max_dd:
                    max_dd = dd
                    trough_date = date
        else:
            if in_drawdown:
                in_drawdown = False
                recovery_date = date
                duration = (recovery_date - peak_date).days
                periods.append({
                    'Peak Date': peak_date,
                    'Trough Date': trough_date,
                    'Recovery Date': recovery_date,
                    'Max Drawdown (%)': round(abs(max_dd) * 100.0, 2),
                    'Duration (Days)': duration
                })
        prev_date = date

    if in_drawdown:
        recovery_date = equity_series.index[-1]
        duration = (recovery_date - peak_date).days
        periods.append({
            'Peak Date': peak_date,
            'Trough Date': trough_date,
            'Recovery Date': pd.NaT,
            'Max Drawdown (%)': round(abs(max_dd) * 100.0, 2),
            'Duration (Days)': duration
        })

    df_dd = pd.DataFrame(periods)
    if len(df_dd) > 0:
        df_dd = df_dd.sort_values(by='Max Drawdown (%)', ascending=False).head(top_n).reset_index(drop=True)
    return df_dd


def calculate_comprehensive_metrics(equity_series, trades_df, initial_capital, data_index, benchmark_equity=None):
    """
    Tính toán toàn bộ các chỉ số định lượng tài chính theo chuẩn mực Hedge Fund:
    Lợi nhuận, CAGR, Volatility, Downside Vol, Sharpe, Sortino, Calmar, Omega,
    MDD, VaR 95%, CVaR 95%, Winning Rate, Profit Factor, Expectancy, Payoff Ratio...
    """
    initial_cap = float(initial_capital)
    end_equity = float(equity_series.iloc[-1])
    total_return_pct = ((end_equity / initial_cap) - 1.0) * 100.0
    net_profit = end_equity - initial_cap

    # 1. Lợi nhuận hàng ngày (Daily Returns)
    daily_returns = equity_series.pct_change().dropna()
    num_sessions = len(equity_series)
    years = max(num_sessions / 252.0, 0.01)

    # CAGR (Compound Annual Growth Rate)
    cagr_pct = (((end_equity / initial_cap) ** (1.0 / years)) - 1.0) * 100.0 if end_equity > 0 else -100.0

    # 2. Rủi ro & Biến động (Volatility & Downside Risk)
    ann_volatility_pct = (daily_returns.std() * np.sqrt(252.0) * 100.0) if len(daily_returns) > 0 else 0.0
    
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        downside_dev_pct = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252.0) * 100.0
    else:
        downside_dev_pct = 0.0

    # 3. Mức Sụt Giảm Tối Đa (Max Drawdown - MDD)
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown_pct = abs(float(drawdown.min())) * 100.0
    avg_drawdown_pct = abs(float(drawdown[drawdown < 0].mean())) * 100.0 if (drawdown < 0).any() else 0.0

    # Max Drawdown Duration (Số ngày chìm trong sụt giảm dài nhất)
    is_underwater = drawdown < 0
    dd_durations = []
    current_dur = 0
    for val in is_underwater:
        if val:
            current_dur += 1
        else:
            if current_dur > 0:
                dd_durations.append(current_dur)
            current_dur = 0
    if current_dur > 0:
        dd_durations.append(current_dur)
    max_dd_duration_days = max(dd_durations) if dd_durations else 0

    # 4. Tỷ Số Hiệu Quả Điều Chỉnh Rủi Ro (Risk-Adjusted Performance Ratios)
    # Sharpe Ratio
    if len(daily_returns) > 0 and daily_returns.std() > 1e-9:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252.0)
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio = CAGR / Downside Deviation
    if downside_dev_pct > 1e-9:
        sortino_ratio = cagr_pct / downside_dev_pct
    else:
        sortino_ratio = 0.0

    # Calmar Ratio = CAGR / Max Drawdown
    if max_drawdown_pct > 1e-9:
        calmar_ratio = cagr_pct / max_drawdown_pct
    else:
        calmar_ratio = 0.0

    # Omega Ratio
    pos_ret_sum = daily_returns[daily_returns > 0].sum()
    neg_ret_sum = abs(daily_returns[daily_returns < 0].sum())
    omega_ratio = (pos_ret_sum / neg_ret_sum) if neg_ret_sum > 0 else (np.inf if pos_ret_sum > 0 else 0.0)

    # 5. Giá trị chịu rủi ro (Value at Risk - VaR 95% & CVaR 95%)
    if len(daily_returns) > 0:
        var_95_pct = abs(float(np.percentile(daily_returns, 5))) * 100.0
        cvar_95_pct = abs(float(daily_returns[daily_returns <= -var_95_pct / 100.0].mean())) * 100.0 if (daily_returns <= -var_95_pct / 100.0).any() else var_95_pct
    else:
        var_95_pct = 0.0
        cvar_95_pct = 0.0

    # 6. Thống Kê Giao Dịch & Tỷ Lệ Thắng (Trade Statistics & Winning Rate)
    num_trades = len(trades_df)
    if num_trades > 0:
        winning_trades = trades_df[trades_df['PnL (%)'] > 0]
        losing_trades = trades_df[trades_df['PnL (%)'] <= 0]
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        winning_rate_pct = (win_count / num_trades) * 100.0
        
        total_win_vnd = winning_trades['PnL (VND)'].sum()
        total_loss_vnd = abs(losing_trades['PnL (VND)'].sum())
        profit_factor = (total_win_vnd / total_loss_vnd) if total_loss_vnd > 0 else (np.inf if total_win_vnd > 0 else 0.0)

        avg_win_pct = winning_trades['PnL (%)'].mean() if win_count > 0 else 0.0
        avg_loss_pct = abs(losing_trades['PnL (%)'].mean()) if loss_count > 0 else 0.0
        payoff_ratio = (avg_win_pct / avg_loss_pct) if avg_loss_pct > 0 else (np.inf if avg_win_pct > 0 else 0.0)

        best_trade_pct = trades_df['PnL (%)'].max()
        worst_trade_pct = trades_df['PnL (%)'].min()

        # Kỳ vọng toán học mỗi lệnh (Expectancy)
        win_prob = win_count / num_trades
        loss_prob = loss_count / num_trades
        expectancy_pct = (win_prob * avg_win_pct) - (loss_prob * avg_loss_pct)
        expectancy_vnd = trades_df['PnL (VND)'].mean()

        avg_holding_days = trades_df['Holding Days'].mean()
        avg_win_holding_days = winning_trades['Holding Days'].mean() if win_count > 0 else 0.0
        avg_loss_holding_days = losing_trades['Holding Days'].mean() if loss_count > 0 else 0.0

        pnl_signs = (trades_df['PnL (%)'] > 0).astype(int)
        max_consec_wins = 0
        max_consec_losses = 0
        cur_wins = 0
        cur_losses = 0
        for sign in pnl_signs:
            if sign == 1:
                cur_wins += 1
                cur_losses = 0
                max_consec_wins = max(max_consec_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_consec_losses = max(max_consec_losses, cur_losses)
    else:
        winning_rate_pct = 0.0
        win_count = 0
        loss_count = 0
        profit_factor = 0.0
        avg_win_pct = 0.0
        avg_loss_pct = 0.0
        payoff_ratio = 0.0
        best_trade_pct = 0.0
        worst_trade_pct = 0.0
        expectancy_pct = 0.0
        expectancy_vnd = 0.0
        avg_holding_days = 0.0
        avg_win_holding_days = 0.0
        avg_loss_holding_days = 0.0
        max_consec_wins = 0
        max_consec_losses = 0

    # 7. Lợi Nhuận Từng Năm (Yearly Returns Breakdown)
    yearly_returns = {}
    years_list = sorted(list(data_index.year.unique()))
    for yr in years_list:
        yr_mask = (equity_series.index.year == yr)
        yr_eq = equity_series.loc[yr_mask]
        if len(yr_eq) > 1:
            yr_ret = ((yr_eq.iloc[-1] / yr_eq.iloc[0]) - 1.0) * 100.0
            yearly_returns[yr] = yr_ret

    # 8. Lợi nhuận vượt trội so với Benchmark (Alpha / Excess Return)
    if benchmark_equity is not None:
        bnh_ret = ((benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1.0) * 100.0
        excess_return_pct = total_return_pct - bnh_ret
    else:
        excess_return_pct = 0.0

    return {
        'Total Return [%]': total_return_pct,
        'Net Profit [VND]': net_profit,
        'CAGR [%]': cagr_pct,
        'Excess Return [%]': excess_return_pct,
        'Annual Volatility [%]': ann_volatility_pct,
        'Downside Deviation [%]': downside_dev_pct,
        'Max Drawdown [%]': max_drawdown_pct,
        'Average Drawdown [%]': avg_drawdown_pct,
        'Max Drawdown Duration [Days]': max_dd_duration_days,
        'VaR (95%) [%]': var_95_pct,
        'CVaR (95%) [%]': cvar_95_pct,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Omega Ratio': omega_ratio,
        'Total Closed Trades': num_trades,
        'Winning Trades': win_count,
        'Losing Trades': loss_count,
        'Winning Rate [%]': winning_rate_pct,
        'Profit Factor': profit_factor,
        'Payoff Ratio': payoff_ratio,
        'Expectancy [%]': expectancy_pct,
        'Expectancy [VND]': expectancy_vnd,
        'Best Trade [%]': best_trade_pct,
        'Worst Trade [%]': worst_trade_pct,
        'Avg Win [%]': avg_win_pct,
        'Avg Loss [%]': avg_loss_pct,
        'Max Consecutive Wins': max_consec_wins,
        'Max Consecutive Losses': max_consec_losses,
        'Avg Holding Days': avg_holding_days,
        'Avg Win Holding Days': avg_win_holding_days,
        'Avg Loss Holding Days': avg_loss_holding_days,
        'Yearly Returns': yearly_returns,
        'Drawdown Series': drawdown
    }


def run_backtest_engine(data, entries, exits, initial_capital=100_000_000, 
                        fees=0.002, slippage=0.001, sl_stop=0.07, benchmark_equity=None):
    """
    Bộ máy mô phỏng giao dịch định lượng chuẩn xác có hạch toán vị thế (Position Accounting).
    """
    n = len(data)
    close = data['Close'].values
    dates = data.index

    positions = np.zeros(n, dtype=int)
    buy_orders = np.zeros(n, dtype=bool)
    sell_orders = np.zeros(n, dtype=bool)
    cash = np.zeros(n, dtype=float)
    holdings = np.zeros(n, dtype=float)
    equity = np.zeros(n, dtype=float)

    current_cash = float(initial_capital)
    current_shares = 0.0
    current_position = 0
    entry_price = 0.0
    entry_date = None

    trades = []

    for i in range(n):
        c_price = close[i]
        is_entry = entries.iloc[i]
        is_exit = exits.iloc[i]

        sl_triggered = False
        if current_position == 1 and sl_stop is not None and sl_stop > 0:
            if c_price <= entry_price * (1.0 - sl_stop):
                sl_triggered = True

        if current_position == 0:
            if is_entry:
                eff_buy_price = c_price * (1.0 + slippage)
                cost_per_share = eff_buy_price * (1.0 + fees)
                if current_cash > cost_per_share:
                    current_shares = current_cash / cost_per_share
                    current_cash = 0.0
                    current_position = 1
                    entry_price = eff_buy_price
                    entry_date = dates[i]
                    buy_orders[i] = True
        elif current_position == 1:
            if is_exit or sl_triggered:
                eff_sell_price = c_price * (1.0 - slippage)
                gross_revenue = current_shares * eff_sell_price
                net_revenue = gross_revenue * (1.0 - fees)
                
                pnl_pct = (eff_sell_price / entry_price - 1.0) * 100.0 - (fees * 2 * 100.0)
                pnl_val = net_revenue - (entry_price * current_shares * (1.0 + fees))
                holding_days = (dates[i] - entry_date).days

                trades.append({
                    'Order ID': len(trades) + 1,
                    'Entry Date': entry_date,
                    'Entry Price': round(entry_price, 1),
                    'Exit Date': dates[i],
                    'Exit Price': round(eff_sell_price, 1),
                    'Holding Days': holding_days,
                    'PnL (%)': round(pnl_pct, 2),
                    'PnL (VND)': round(pnl_val, 0),
                    'Exit Reason': 'Stop Loss (7%)' if sl_triggered else 'Tín hiệu EMA cắt'
                })

                current_cash = net_revenue
                current_shares = 0.0
                current_position = 0
                entry_price = 0.0
                entry_date = None
                sell_orders[i] = True

        positions[i] = current_position
        cash[i] = current_cash
        holdings[i] = current_shares
        equity[i] = current_cash + (current_shares * c_price)

    equity_series = pd.Series(equity, index=dates)
    positions_series = pd.Series(positions, index=dates)
    buy_series = pd.Series(buy_orders, index=dates)
    sell_series = pd.Series(sell_orders, index=dates)
    trades_df = pd.DataFrame(trades)

    metrics = calculate_comprehensive_metrics(equity_series, trades_df, initial_capital, dates, benchmark_equity)
    metrics['Position Coverage [%]'] = (positions_series.sum() / len(positions_series)) * 100.0
    top_drawdowns_df = calculate_top_drawdowns(equity_series, top_n=5)

    return {
        'metrics': metrics,
        'equity_curve': equity_series,
        'drawdown': metrics['Drawdown Series'],
        'positions': positions_series,
        'buy_orders': buy_series,
        'sell_orders': sell_series,
        'trades_df': trades_df,
        'top_drawdowns': top_drawdowns_df
    }


def calculate_benchmark_metrics(data, initial_capital=100_000_000, fees=0.002, slippage=0.001):
    """Tính toán hiệu suất của Chiến lược Mua & Nắm giữ (Buy & Hold Benchmark)."""
    close = data['Close']
    first_price = close.iloc[0] * (1.0 + slippage) * (1.0 + fees)
    shares = initial_capital / first_price
    equity_series = shares * close * (1.0 - slippage) * (1.0 - fees)
    
    empty_trades = pd.DataFrame()
    metrics = calculate_comprehensive_metrics(equity_series, empty_trades, initial_capital, data.index)
    metrics['Position Coverage [%]'] = 100.0
    top_dd = calculate_top_drawdowns(equity_series, top_n=5)

    return {
        'metrics': metrics,
        'equity_curve': equity_series,
        'drawdown': metrics['Drawdown Series'],
        'top_drawdowns': top_dd
    }

# ==============================================================================
# GIAO DIỆN THANH BÊN (SIDEBAR CONTROLS)
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=64)
    st.title("Bảng Điều Khiển")
    st.caption("Chiến lược EMA & OBV | Quản trị Vị thế")
    st.markdown("---")

    # 1. Nguồn Dữ Liệu
    st.subheader("1. Dữ Liệu Giá")
    data_source = st.radio("Nguồn nạp dữ liệu:", ["Dữ liệu mẫu ACB.csv (10 năm)", "Tải lên file CSV tùy chỉnh"], index=0)
    uploaded_file = None
    if data_source == "Tải lên file CSV tùy chỉnh":
        uploaded_file = st.file_uploader("Chọn file CSV (Cần cột: Date, Close, Volume)", type=["csv"])

    df_raw = load_data(uploaded_file)
    if df_raw is None:
        st.stop()

    st.success(f"Đã nạp {len(df_raw)} phiên: {df_raw.index.min().strftime('%d/%m/%Y')} ➔ {df_raw.index.max().strftime('%d/%m/%Y')}")

    # 2. Phân Chia Tập Dữ Liệu Train / Test
    st.subheader("2. Phân Chia Dữ Liệu")
    years_available = sorted(list(df_raw.index.year.unique()))
    
    col_tr, col_te = st.columns(2)
    with col_tr:
        train_start = st.selectbox("Train từ năm", years_available, index=0)
    with col_te:
        train_end_default_idx = years_available.index(2020) if 2020 in years_available else len(years_available) - 4
        train_end = st.selectbox("Train đến năm", years_available, index=train_end_default_idx)

    test_start = train_end + 1
    test_end = years_available[-1]
    st.caption(f"📌 **Tập Train:** {train_start} – {train_end} | **Tập Test:** {test_start} – {test_end}")

    # 3. Tham Số Chiến Lược
    st.subheader("3. Cấu Hình Tham Số")
    preset_mode = st.radio(
        "Chế độ cấu hình tham số:",
        ["🏆 Tối ưu Nghiên cứu (Best Preset)", "📌 Mặc định (Default Preset)", "🛠️ Tự điều chỉnh (Custom)"],
        index=0
    )

    if preset_mode == "🏆 Tối ưu Nghiên cứu (Best Preset)":
        ema_param = 36
        obv_param = 20
        st.info("💡 **Bộ tham số tối ưu (Nghiên cứu):**\n- EMA Period = 36\n- OBV Slope Period = 20")
    elif preset_mode == "📌 Mặc định (Default Preset)":
        ema_param = 20
        obv_param = 3
        st.info("💡 **Bộ tham số mặc định:**\n- EMA Period = 20\n- OBV Slope Period = 3")
    else:
        ema_param = st.slider("Chu kỳ EMA (EMA Period):", min_value=5, max_value=80, value=36, step=1)
        obv_param = st.slider("Chu kỳ Độ dốc OBV (OBV Slope):", min_value=1, max_value=30, value=20, step=1)

    # 4. Quản Trị Rủi Ro & Chi Phí
    st.subheader("4. Quản Trị Rủi Ro & Phí")
    with st.expander("⚙️ Cài đặt Quản trị Vị thế & Chi phí", expanded=False):
        initial_capital = st.number_input("Vốn ban đầu (VND):", value=100_000_000, step=10_000_000, format="%d")
        stop_loss_pct = st.slider("Mức Cắt lỗ Cứng (Stop Loss %):", min_value=0.0, max_value=15.0, value=7.0, step=0.5) / 100.0
        fee_pct = st.number_input("Phí giao dịch mỗi chiều (%):", value=0.2, step=0.05) / 100.0
        slippage_pct = st.number_input("Trượt giá mỗi chiều (%):", value=0.1, step=0.05) / 100.0
        min_trades_filter = st.number_input("Số lệnh tối thiểu (MIN_TRADES):", value=5, min_value=1, step=1)

    st.markdown("---")
    st.caption("Hệ thống Phân tích Định lượng Chuyên sâu | Streamlit Cloud")

# ==============================================================================
# XỬ LÝ DỮ LIỆU & TÍNH TOÁN TOÀN CỤC
# ==============================================================================

df_calculated = calculate_indicators(df_raw, ema_period=ema_param, obv_slope_period=obv_param)

train_mask = (df_calculated.index.year >= train_start) & (df_calculated.index.year <= train_end)
test_mask = (df_calculated.index.year >= test_start) & (df_calculated.index.year <= test_end)

train_df = df_calculated.loc[train_mask].copy()
test_df = df_calculated.loc[test_mask].copy()

signals_train = generate_all_signals(train_df, ema_param, obv_param)
signals_test = generate_all_signals(test_df, ema_param, obv_param) if len(test_df) > 0 else None
signals_full = generate_all_signals(df_calculated, ema_param, obv_param)

bnh_train = calculate_benchmark_metrics(train_df, initial_capital, fee_pct, slippage_pct)
bnh_test = calculate_benchmark_metrics(test_df, initial_capital, fee_pct, slippage_pct) if len(test_df) > 0 else None
bnh_full = calculate_benchmark_metrics(df_calculated, initial_capital, fee_pct, slippage_pct)

bt_comb_train = run_backtest_engine(train_df, signals_train['Combined'][0], signals_train['Combined'][1],
                                    initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_train['equity_curve'])
bt_ema_train = run_backtest_engine(train_df, signals_train['EMA'][0], signals_train['EMA'][1],
                                   initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_train['equity_curve'])
bt_obv_train = run_backtest_engine(train_df, signals_train['OBV'][0], signals_train['OBV'][1],
                                   initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_train['equity_curve'])

if len(test_df) > 0:
    bt_comb_test = run_backtest_engine(test_df, signals_test['Combined'][0], signals_test['Combined'][1],
                                       initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_test['equity_curve'])
    bt_ema_test = run_backtest_engine(test_df, signals_test['EMA'][0], signals_test['EMA'][1],
                                      initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_test['equity_curve'])
    bt_obv_test = run_backtest_engine(test_df, signals_test['OBV'][0], signals_test['OBV'][1],
                                      initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_test['equity_curve'])
else:
    bt_comb_test = bt_ema_test = bt_obv_test = None

bt_comb_full = run_backtest_engine(df_calculated, signals_full['Combined'][0], signals_full['Combined'][1],
                                   initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_full['equity_curve'])
bt_ema_full = run_backtest_engine(df_calculated, signals_full['EMA'][0], signals_full['EMA'][1],
                                  initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_full['equity_curve'])
bt_obv_full = run_backtest_engine(df_calculated, signals_full['OBV'][0], signals_full['OBV'][1],
                                  initial_capital, fee_pct, slippage_pct, stop_loss_pct, bnh_full['equity_curve'])

# ==============================================================================
# TIÊU ĐỀ TRANG & BADGES
# ==============================================================================
st.markdown('<div class="main-title">Hệ Thống Phân Tích & Kiểm Định Định Lượng: EMA & OBV</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Chuyên sâu Phân tích Lợi nhuận (Returns), Tỷ số Calmar, Sortino, Độ biến động (Volatility), MDD và Tỷ lệ thắng (Winning Rate)</div>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-container">
    <span class="tech-badge-primary">📊 Cổ phiếu ACB</span>
    <span class="tech-badge">🎯 Tỷ số Calmar & Sortino</span>
    <span class="tech-badge">📉 Maximum Drawdown (MDD)</span>
    <span class="tech-badge">🌊 Phân tích Biến động (Volatility)</span>
    <span class="tech-badge">🏆 Tỷ lệ thắng & Profit Factor</span>
    <span class="tech-badge">🛡️ Hạch toán Vị thế & Stop Loss 7%</span>
    <span class="tech-badge">⚡ Không Look-ahead Bias [shift(1)]</span>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# BỐ CỤC 5 TABS CHÍNH
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Tín Hiệu & Biểu Đồ Kỹ Thuật",
    "🧪 Kiểm Định Hiệu Quả & Phân Tích Chuyên Sâu",
    "📋 Quản Lý Vị Thế & Nhật Ký Lệnh",
    "🎯 Tối Ưu Hóa Tham Số (Hyperopt)",
    "📖 Phương Pháp Luận & Khắc Phục Lỗi"
])

# ==============================================================================
# TAB 1: TÍN HIỆU & BIỂU ĐỒ KỸ THUẬT
# ==============================================================================
with tab1:
    st.subheader("Trạng Thái Vị Thế & Khuyến Nghị Phiên Mới Nhất")

    last_date = df_calculated.index[-1].strftime('%d/%m/%Y')
    last_close = df_calculated['Close'].iloc[-1]
    prev_close = df_calculated['Close'].iloc[-2]
    price_change = last_close - prev_close
    price_change_pct = (price_change / prev_close) * 100.0

    last_ema = df_calculated['EMA'].iloc[-1]
    last_obv_slope = df_calculated['OBV_Slope'].iloc[-1]
    last_position = bt_comb_full['positions'].iloc[-1]

    last_entry_signal = signals_full['Combined'][0].iloc[-1]
    last_exit_signal = signals_full['Combined'][1].iloc[-1]

    if last_position == 1:
        if last_exit_signal:
            rec_action = "🔴 KHUYẾN NGHỊ BÁN (SELL)"
            rec_detail = "Giá đã giảm xuống dưới đường EMA. Kích hoạt tín hiệu đóng vị thế bảo toàn vốn."
            box_class = "signal-box-sell"
        else:
            rec_action = "🔵 TIẾP TỤC NẮM GIỮ (HOLD)"
            rec_detail = f"Tài khoản đang giữ cổ phiếu. Giá ({last_close:,.0f}) vẫn duy trì trên đường EMA ({last_ema:,.1f})."
            box_class = "signal-box-hold"
    else:
        if last_entry_signal:
            rec_action = "🟢 KHUYẾN NGHỊ MUA (BUY)"
            rec_detail = f"Điều kiện thỏa mãn: Giá vượt trên EMA ({last_ema:,.1f}) VÀ Độ dốc OBV dương ({last_obv_slope:+,.0f}). Mở vị thế mới."
            box_class = "signal-box-buy"
        else:
            rec_action = "⚪ ĐỨNG NGOÀI THỊ TRƯỜNG (STAND ASIDE / CASH)"
            rec_detail = "Chưa hội đủ tín hiệu đồng thuận từ cả EMA và OBV. Giữ tiền mặt để quản trị rủi ro."
            box_class = "signal-box-cash"

    st.markdown(f"""
    <div class="{box_class}">
        <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 4px;">{rec_action} (Phiên ngày {last_date})</div>
        <div style="font-size: 0.95rem; margin-bottom: 8px;">{rec_detail}</div>
        <div style="font-size: 0.85rem; color: #4B5563;">
            <b>Trạng thái tài khoản:</b> {'Đang nắm giữ cổ phiếu (Long 100%)' if last_position == 1 else 'Đang nắm giữ tiền mặt (Cash 100%)'} |
            <b>Giá đóng cửa:</b> {last_close:,.0f} VND ({price_change_pct:+.2f}%) |
            <b>Đường EMA({ema_param}):</b> {last_ema:,.1f} |
            <b>Độ dốc OBV({obv_param}):</b> {last_obv_slope:+,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 📈 Biểu Đồ Kỹ Thuật Tương Tác (Giá, EMA, Tín Hiệu & OBV)")
    col_chart_filter1, col_chart_filter2 = st.columns([2, 1])
    with col_chart_filter1:
        timeframe_choice = st.radio(
            "Khoảng thời gian hiển thị:",
            ["Tất cả (2014 - 2023)", "Tập Test (2021 - 2023)", "1 Năm gần nhất", "Tùy chọn ngày"],
            horizontal=True
        )
    
    if timeframe_choice == "Tất cả (2014 - 2023)":
        plot_df = df_calculated.copy()
    elif timeframe_choice == "Tập Test (2021 - 2023)":
        plot_df = test_df.copy()
    elif timeframe_choice == "1 Năm gần nhất":
        one_year_ago = df_calculated.index[-1] - pd.DateOffset(years=1)
        plot_df = df_calculated.loc[df_calculated.index >= one_year_ago].copy()
    else:
        with col_chart_filter2:
            date_range = st.date_input("Chọn khoảng ngày:", [df_calculated.index[-250].date(), df_calculated.index[-1].date()])
        if len(date_range) == 2:
            plot_df = df_calculated.loc[(df_calculated.index.date >= date_range[0]) & (df_calculated.index.date <= date_range[1])].copy()
        else:
            plot_df = df_calculated.copy()

    plot_buys = bt_comb_full['buy_orders'].loc[plot_df.index]
    plot_sells = bt_comb_full['sell_orders'].loc[plot_df.index]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
        subplot_titles=(f"Đồ thị Giá ACB & Đường EMA({ema_param}) kèm Điểm Khớp Lệnh", f"Chỉ báo On-Balance Volume (OBV) & Độ Dốc ({obv_param} phiên)")
    )

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines', name='Giá Đóng Cửa',
                   line=dict(color='#1E3A8A', width=1.5)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['EMA'], mode='lines', name=f'EMA ({ema_param})',
                   line=dict(color='#F59E0B', width=1.8, dash='dash')),
        row=1, col=1
    )

    buy_points = plot_df.loc[plot_buys]
    if len(buy_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_points.index, y=buy_points['Close'] * 0.98,
                mode='markers', name='Điểm MUA',
                marker=dict(symbol='triangle-up', size=11, color='#10B981', line=dict(width=1, color='#047857')),
                text=[f"MUA: {p:,.0f} VND" for p in buy_points['Close']],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )

    sell_points = plot_df.loc[plot_sells]
    if len(sell_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_points.index, y=sell_points['Close'] * 1.02,
                mode='markers', name='Điểm BÁN',
                marker=dict(symbol='triangle-down', size=11, color='#EF4444', line=dict(width=1, color='#B91C1C')),
                text=[f"BÁN: {p:,.0f} VND" for p in sell_points['Close']],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df['OBV'], mode='lines', name='OBV',
                   line=dict(color='#6366F1', width=1.2)),
        row=2, col=1
    )

    colors = ['#10B981' if val > 0 else '#EF4444' for val in plot_df['OBV_Slope']]
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df['OBV_Slope'], name=f'Độ dốc OBV ({obv_param})',
               marker=dict(color=colors, opacity=0.75)),
        row=2, col=1
    )

    fig.update_layout(
        height=620,
        margin=dict(l=40, r=40, t=40, b=30),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    fig.update_yaxes(title_text="Giá (VND)", row=1, col=1)
    fig.update_yaxes(title_text="Độ dốc / Khối lượng", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 📋 Bảng Chi Tiết 10 Phiên Giao Dịch Gần Nhất")
    recent_df = plot_df.tail(10).copy()
    recent_display = pd.DataFrame({
        'Ngày': recent_df.index.strftime('%d/%m/%Y'),
        'Giá Đóng Cửa (VND)': recent_df['Close'].apply(lambda x: f"{x:,.0f}"),
        f'EMA ({ema_param})': recent_df['EMA'].apply(lambda x: f"{x:,.1f}"),
        f'Độ Dốc OBV ({obv_param})': recent_df['OBV_Slope'].apply(lambda x: f"{x:+,.0f}"),
        'Tín Hiệu Mua [t-1]': signals_full['Combined'][0].loc[recent_df.index].apply(lambda x: "✅ Có" if x else "—"),
        'Tín Hiệu Bán [t-1]': signals_full['Combined'][1].loc[recent_df.index].apply(lambda x: "🔻 Có" if x else "—"),
        'Vị Thế Nắm Giữ': bt_comb_full['positions'].loc[recent_df.index].apply(lambda x: "Cổ phiếu (1)" if x == 1 else "Tiền mặt (0)"),
        'Khớp Lệnh': [
            "🟢 MUA" if b else ("🔴 BÁN" if s else "—")
            for b, s in zip(bt_comb_full['buy_orders'].loc[recent_df.index], bt_comb_full['sell_orders'].loc[recent_df.index])
        ]
    })
    st.dataframe(recent_display, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 2: KIỂM ĐỊNH HIỆU QUẢ & PHÂN TÍCH CHUYÊN SÂU
# ==============================================================================
with tab2:
    st.subheader("Báo Cáo Phân Tích Định Lượng Chuyên Sâu (Quantitative Performance Tear-Sheet)")

    eval_dataset = st.radio(
        "Chọn phân đoạn dữ liệu kiểm định:",
        ["Tập Huấn Luyện (Train: 2014 - 2020)", "Tập Kiểm Định Độc Lập (Test: 2021 - 2023)", "Toàn Bộ Lịch Sử (Full: 2014 - 2023)"],
        horizontal=True
    )

    if "Train" in eval_dataset:
        act_comb = bt_comb_train
        act_ema = bt_ema_train
        act_obv = bt_obv_train
        act_bnh = bnh_train
        act_df = train_df
        set_name = "Tập Train (2014 - 2020)"
    elif "Test" in eval_dataset:
        act_comb = bt_comb_test
        act_ema = bt_ema_test
        act_obv = bt_obv_test
        act_bnh = bnh_test
        act_df = test_df
        set_name = "Tập Test (2021 - 2023)"
    else:
        act_comb = bt_comb_full
        act_ema = bt_ema_full
        act_obv = bt_obv_full
        act_bnh = bnh_full
        act_df = df_calculated
        set_name = "Toàn Bộ Lịch Sử (2014 - 2023)"

    mc = act_comb['metrics']
    mb = act_bnh['metrics']

    # --------------------------------------------------------------------------
    # 6 KHỐI METRICS KPI VÀNG (TOP TIER METRICS)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">🎯 Các Chỉ Số Trọng Tâm (Key Performance Indicators)</div>', unsafe_allow_html=True)
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Tổng Lợi Nhuận</span>
                <span class="kpi-icon">💰</span>
            </div>
            <div class="kpi-value" style="color: {'#059669' if mc['Total Return [%]'] >= 0 else '#DC2626'};">
                {mc['Total Return [%]']:+,.1f}%
            </div>
            <div class="kpi-sub">CAGR: <b>{mc['CAGR [%]']:+,.1f}%/năm</b><br>B&H: {mb['Total Return [%]']:+,.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        calmar_val = mc['Calmar Ratio']
        calmar_color = '#059669' if calmar_val >= 1.0 else ('#D97706' if calmar_val >= 0.5 else '#DC2626')
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Calmar Ratio</span>
                <span class="kpi-icon">🛡️</span>
            </div>
            <div class="kpi-value" style="color: {calmar_color};">
                {calmar_val:.2f}
            </div>
            <div class="kpi-sub">CAGR / MDD<br>B&H: {mb['Calmar Ratio']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        sortino_val = mc['Sortino Ratio']
        sortino_color = '#059669' if sortino_val >= 1.5 else ('#D97706' if sortino_val >= 0.8 else '#DC2626')
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Sortino Ratio</span>
                <span class="kpi-icon">⚖️</span>
            </div>
            <div class="kpi-value" style="color: {sortino_color};">
                {sortino_val:.2f}
            </div>
            <div class="kpi-sub">Sharpe: <b>{mc['Sharpe Ratio']:.2f}</b><br>B&H Sortino: {mb['Sortino Ratio']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Độ Biến Động</span>
                <span class="kpi-icon">🌊</span>
            </div>
            <div class="kpi-value" style="color: #2563EB;">
                {mc['Annual Volatility [%]']:.1f}%
            </div>
            <div class="kpi-sub">Downside Vol: <b>{mc['Downside Deviation [%]']:.1f}%</b><br>B&H Vol: {mb['Annual Volatility [%]']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        mdd_color = '#059669' if mc['Max Drawdown [%]'] < 25.0 else ('#D97706' if mc['Max Drawdown [%]'] < 40.0 else '#DC2626')
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Max Drawdown (MDD)</span>
                <span class="kpi-icon">📉</span>
            </div>
            <div class="kpi-value" style="color: {mdd_color};">
                -{mc['Max Drawdown [%]']:.1f}%
            </div>
            <div class="kpi-sub">Thời gian sụt: <b>{mc['Max Drawdown Duration [Days]']} ngày</b><br>B&H MDD: -{mb['Max Drawdown [%]']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col6:
        win_color = '#059669' if mc['Winning Rate [%]'] >= 40.0 else '#D97706'
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">
                <span class="kpi-title">Tỷ Lệ Thắng (Win Rate)</span>
                <span class="kpi-icon">🏆</span>
            </div>
            <div class="kpi-value" style="color: {win_color};">
                {mc['Winning Rate [%]']:.1f}%
            </div>
            <div class="kpi-sub">Profit Factor: <b>{mc['Profit Factor']:.2f}</b><br>Số lệnh: {mc['Total Closed Trades']}</div>
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # BIỂU ĐỒ EQUITY CURVE & UNDERWATER DRAWDOWN
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">📈 Tăng Trưởng Tài Sản & Đường Sụt Giảm (Equity & Drawdown)</div>', unsafe_allow_html=True)
    
    col_chart_l, col_chart_r = st.columns([7, 3])
    with col_chart_l:
        fig_dual = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.72, 0.28],
            subplot_titles=(f"Đường Tăng Trưởng Vốn (Equity Curve) — {set_name}", "Mức Sụt Giảm Từ Đỉnh (Underwater Drawdown %)")
        )

        fig_dual.add_trace(go.Scatter(x=act_comb['equity_curve'].index, y=act_comb['equity_curve'],
                                      mode='lines', name='⭐ EMA + OBV Kết Hợp',
                                      line=dict(color='#059669', width=2.5)), row=1, col=1)
        fig_dual.add_trace(go.Scatter(x=act_ema['equity_curve'].index, y=act_ema['equity_curve'],
                                      mode='lines', name='EMA Riêng Lẻ',
                                      line=dict(color='#2563EB', width=1.5, dash='dot')), row=1, col=1)
        fig_dual.add_trace(go.Scatter(x=act_obv['equity_curve'].index, y=act_obv['equity_curve'],
                                      mode='lines', name='OBV Riêng Lẻ',
                                      line=dict(color='#D97706', width=1.5, dash='dash')), row=1, col=1)
        fig_dual.add_trace(go.Scatter(x=act_bnh['equity_curve'].index, y=act_bnh['equity_curve'],
                                      mode='lines', name='Buy & Hold (ACB Benchmark)',
                                      line=dict(color='#94A3B8', width=1.5)), row=1, col=1)

        fig_dual.add_trace(go.Scatter(x=act_comb['drawdown'].index, y=act_comb['drawdown'] * 100,
                                      mode='lines', fill='tozeroy', name='DD Chiến lược (%)',
                                      line=dict(color='#DC2626', width=1), fillcolor='rgba(220, 38, 38, 0.2)'), row=2, col=1)
        fig_dual.add_trace(go.Scatter(x=act_bnh['drawdown'].index, y=act_bnh['drawdown'] * 100,
                                      mode='lines', name='DD Benchmark (%)',
                                      line=dict(color='#94A3B8', width=1, dash='dot')), row=2, col=1)

        fig_dual.update_layout(
            height=540,
            margin=dict(l=40, r=40, t=30, b=20),
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_dual.update_yaxes(title_text="Vốn (VND)", row=1, col=1)
        fig_dual.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        st.plotly_chart(fig_dual, use_container_width=True)

    with col_chart_r:
        st.markdown("###### 🕸️ Radar Đánh Giá Năng Lực Chiến Lược")
        radar_categories = ['Lợi Nhuận (CAGR)', 'Kiểm Soát Rủi Ro', 'Tỷ Số Calmar', 'Tỷ Số Sortino', 'Tỷ Lệ Thắng', 'Profit Factor']
        
        c_score_ret = np.clip((mc['CAGR [%]'] + 20) / 60 * 100, 0, 100)
        c_score_mdd = np.clip((100 - mc['Max Drawdown [%]']), 0, 100)
        c_score_cal = np.clip(mc['Calmar Ratio'] / 1.5 * 100, 0, 100)
        c_score_sor = np.clip(mc['Sortino Ratio'] / 2.0 * 100, 0, 100)
        c_score_win = np.clip(mc['Winning Rate [%]'] / 60 * 100, 0, 100)
        c_score_pf = np.clip(mc['Profit Factor'] / 2.5 * 100, 0, 100)
        
        b_score_ret = np.clip((mb['CAGR [%]'] + 20) / 60 * 100, 0, 100)
        b_score_mdd = np.clip((100 - mb['Max Drawdown [%]']), 0, 100)
        b_score_cal = np.clip(mb['Calmar Ratio'] / 1.5 * 100, 0, 100)
        b_score_sor = np.clip(mb['Sortino Ratio'] / 2.0 * 100, 0, 100)
        b_score_win = 50.0
        b_score_pf = 50.0

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[c_score_ret, c_score_mdd, c_score_cal, c_score_sor, c_score_win, c_score_pf],
            theta=radar_categories,
            fill='toself',
            name='EMA + OBV',
            line=dict(color='#059669', width=2),
            fillcolor='rgba(5, 150, 105, 0.25)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[b_score_ret, b_score_mdd, b_score_cal, b_score_sor, b_score_win, b_score_pf],
            theta=radar_categories,
            fill='toself',
            name='Benchmark',
            line=dict(color='#94A3B8', width=1.5, dash='dot'),
            fillcolor='rgba(148, 163, 184, 0.15)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            height=460,
            margin=dict(l=40, r=40, t=30, b=50),
            template="plotly_white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --------------------------------------------------------------------------
    # BẢNG MA TRẬN PHÂN TÍCH ĐỊNH LƯỢNG TOÀN DIỆN (TEAR-SHEET MATRIX)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">📑 Bảng Ma Trận Phân Tích Định Lượng Toàn Diện (Quantitative Tearsheet)</div>', unsafe_allow_html=True)
    
    m_comb = act_comb['metrics']
    m_ema = act_ema['metrics']
    m_obv = act_obv['metrics']
    m_bnh = act_bnh['metrics']

    tearsheet_sections = [
        ("1. Phân Tích Lợi Nhuận & Hiệu Suất (Returns & Profitability)", [
            ("Tổng Lợi Nhuận Tích Lũy (Total Return)", f"{m_comb['Total Return [%]']:+.2f}%", f"{m_ema['Total Return [%]']:+.2f}%", f"{m_obv['Total Return [%]']:+.2f}%", f"{m_bnh['Total Return [%]']:+.2f}%"),
            ("Lợi Nhuận Ròng Tuyệt Đối (Net Profit)", f"{m_comb['Net Profit [VND]']:+,.0f} đ", f"{m_ema['Net Profit [VND]']:+,.0f} đ", f"{m_obv['Net Profit [VND]']:+,.0f} đ", f"{m_bnh['Net Profit [VND]']:+,.0f} đ"),
            ("Tỷ Suất Sinh Lời Kép Hàng Năm (CAGR)", f"{m_comb['CAGR [%]']:+.2f}%", f"{m_ema['CAGR [%]']:+.2f}%", f"{m_obv['CAGR [%]']:+.2f}%", f"{m_bnh['CAGR [%]']:+.2f}%"),
            ("Lợi Nhuận Vượt Trội so với Benchmark (Alpha)", f"{m_comb['Excess Return [%]']:+.2f}%", f"{m_ema['Excess Return [%]']:+.2f}%", f"{m_obv['Excess Return [%]']:+.2f}%", "—")
        ]),
        ("2. Đo Lường Rủi Ro & Biến Động (Risk & Volatility Analysis)", [
            ("Độ Biến Động Hàng Năm (Annualized Volatility)", f"{m_comb['Annual Volatility [%]']:.2f}%", f"{m_ema['Annual Volatility [%]']:.2f}%", f"{m_obv['Annual Volatility [%]']:.2f}%", f"{m_bnh['Annual Volatility [%]']:.2f}%"),
            ("Độ Lệch Chuẩn Phía Giảm (Downside Deviation)", f"{m_comb['Downside Deviation [%]']:.2f}%", f"{m_ema['Downside Deviation [%]']:.2f}%", f"{m_obv['Downside Deviation [%]']:.2f}%", f"{m_bnh['Downside Deviation [%]']:.2f}%"),
            ("Mức Sụt Giảm Tối Đa (Maximum Drawdown - MDD)", f"-{m_comb['Max Drawdown [%]']:.2f}%", f"-{m_ema['Max Drawdown [%]']:.2f}%", f"-{m_obv['Max Drawdown [%]']:.2f}%", f"-{m_bnh['Max Drawdown [%]']:.2f}%"),
            ("Mức Sụt Giảm Trung Bình (Average Drawdown)", f"-{m_comb['Average Drawdown [%]']:.2f}%", f"-{m_ema['Average Drawdown [%]']:.2f}%", f"-{m_obv['Average Drawdown [%]']:.2f}%", f"-{m_bnh['Average Drawdown [%]']:.2f}%"),
            ("Thời Gian Sụt Giảm Kéo Dài Nhất (Max DD Duration)", f"{m_comb['Max Drawdown Duration [Days]']} ngày", f"{m_ema['Max Drawdown Duration [Days]']} ngày", f"{m_obv['Max Drawdown Duration [Days]']} ngày", f"{m_bnh['Max Drawdown Duration [Days]']} ngày"),
            ("Giá Trị Chịu Rủi Ro Ngày (VaR 95% 1-day)", f"{m_comb['VaR (95%) [%]']:.2f}%", f"{m_ema['VaR (95%) [%]']:.2f}%", f"{m_obv['VaR (95%) [%]']:.2f}%", f"{m_bnh['VaR (95%) [%]']:.2f}%"),
            ("Tổn Thất Kỳ Vọng Đáy (CVaR / Expected Shortfall 95%)", f"{m_comb['CVaR (95%) [%]']:.2f}%", f"{m_ema['CVaR (95%) [%]']:.2f}%", f"{m_obv['CVaR (95%) [%]']:.2f}%", f"{m_bnh['CVaR (95%) [%]']:.2f}%")
        ]),
        ("3. Tỷ Số Hiệu Quả Điều Chỉnh Rủi Ro (Risk-Adjusted Ratios)", [
            ("Sharpe Ratio (Rf = 0%, freq='D')", f"{m_comb['Sharpe Ratio']:.4f}", f"{m_ema['Sharpe Ratio']:.4f}", f"{m_obv['Sharpe Ratio']:.4f}", f"{m_bnh['Sharpe Ratio']:.4f}"),
            ("Sortino Ratio (Đo lường rủi ro sụt giảm)", f"{m_comb['Sortino Ratio']:.4f}", f"{m_ema['Sortino Ratio']:.4f}", f"{m_obv['Sortino Ratio']:.4f}", f"{m_bnh['Sortino Ratio']:.4f}"),
            ("Calmar Ratio (CAGR / Max Drawdown)", f"{m_comb['Calmar Ratio']:.4f}", f"{m_ema['Calmar Ratio']:.4f}", f"{m_obv['Calmar Ratio']:.4f}", f"{m_bnh['Calmar Ratio']:.4f}"),
            ("Omega Ratio (Tỷ lệ xác suất Lãi / Lỗ)", f"{m_comb['Omega Ratio']:.4f}", f"{m_ema['Omega Ratio']:.4f}", f"{m_obv['Omega Ratio']:.4f}", f"{m_bnh['Omega Ratio']:.4f}")
        ]),
        ("4. Thống Kê Giao Dịch & Tỷ Lệ Thắng (Trade Statistics & Winning Rate)", [
            ("Tỷ Lệ Thắng (Winning Rate %)", f"{m_comb['Winning Rate [%]']:.2f}%", f"{m_ema['Winning Rate [%]']:.2f}%", f"{m_obv['Winning Rate [%]']:.2f}%", "—"),
            ("Hệ Số Lợi Nhuận (Profit Factor)", f"{m_comb['Profit Factor']:.2f}", f"{m_ema['Profit Factor']:.2f}", f"{m_obv['Profit Factor']:.2f}", "—"),
            ("Tỷ Số Lãi/Lỗ Bình Quân (Payoff Ratio = Win/Loss)", f"{m_comb['Payoff Ratio']:.2f}", f"{m_ema['Payoff Ratio']:.2f}", f"{m_obv['Payoff Ratio']:.2f}", "—"),
            ("Kỳ Vọng Toán Học Mỗi Lệnh (Trade Expectancy %)", f"{m_comb['Expectancy [%]']:+.2f}%", f"{m_ema['Expectancy [%]']:+.2f}%", f"{m_obv['Expectancy [%]']:+.2f}%", "—"),
            ("Kỳ Vọng Tiền Mỗi Lệnh (Expectancy VND)", f"{m_comb['Expectancy [VND]']:+,.0f} đ", f"{m_ema['Expectancy [VND]']:+,.0f} đ", f"{m_obv['Expectancy [VND]']:+,.0f} đ", "—"),
            ("Tổng Số Lệnh Đã Đóng (Total Closed Trades)", str(m_comb['Total Closed Trades']), str(m_ema['Total Closed Trades']), str(m_obv['Total Closed Trades']), "1"),
            ("Số Lệnh Thắng / Số Lệnh Thua", f"{m_comb['Winning Trades']} / {m_comb['Losing Trades']}", f"{m_ema['Winning Trades']} / {m_ema['Losing Trades']}", f"{m_obv['Winning Trades']} / {m_obv['Losing Trades']}", "—"),
            ("Lệnh Thắng Lớn Nhất (Best Trade %)", f"{m_comb['Best Trade [%]']:+.2f}%", f"{m_ema['Best Trade [%]']:+.2f}%", f"{m_obv['Best Trade [%]']:+.2f}%", "—"),
            ("Lệnh Thua Nặng Nhất (Worst Trade %)", f"{m_comb['Worst Trade [%]']:+.2f}%", f"{m_ema['Worst Trade [%]']:+.2f}%", f"{m_obv['Worst Trade [%]']:+.2f}%", "—"),
            ("Lợi Nhuận Bình Quân Lệnh Thắng", f"{m_comb['Avg Win [%]']:+.2f}%", f"{m_ema['Avg Win [%]']:+.2f}%", f"{m_obv['Avg Win [%]']:+.2f}%", "—"),
            ("Thua Lỗ Bình Quân Lệnh Thua", f"-{m_comb['Avg Loss [%]']:.2f}%", f"-{m_ema['Avg Loss [%]']:.2f}%", f"-{m_obv['Avg Loss [%]']:.2f}%", "—"),
            ("Chuỗi Thắng / Chuỗi Thua Liên Tiếp Tối Đa", f"{m_comb['Max Consecutive Wins']} / {m_comb['Max Consecutive Losses']}", f"{m_ema['Max Consecutive Wins']} / {m_ema['Max Consecutive Losses']}", f"{m_obv['Max Consecutive Wins']} / {m_obv['Max Consecutive Losses']}", "—"),
            ("Thời Gian Giữ Lệnh Bình Quân (Holding Days)", f"{m_comb['Avg Holding Days']:.1f} ngày", f"{m_ema['Avg Holding Days']:.1f} ngày", f"{m_obv['Avg Holding Days']:.1f} ngày", f"{len(act_df)} ngày"),
            ("Tỷ Lệ Thời Gian Giữ Cổ Phiếu (Position Coverage)", f"{m_comb['Position Coverage [%]']:.1f}%", f"{m_ema['Position Coverage [%]']:.1f}%", f"{m_obv['Position Coverage [%]']:.1f}%", "100.0%")
        ])
    ]

    for section_title, rows in tearsheet_sections:
        with st.expander(section_title, expanded=True):
            df_sec = pd.DataFrame(rows, columns=['Chỉ Số Định Lượng', '⭐ EMA + OBV Kết Hợp', 'EMA Riêng Lẻ', 'OBV Riêng Lẻ', 'Buy & Hold (ACB)'])
            st.dataframe(df_sec, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # PHÂN TÍCH LỢI NHUẬN TỪNG NĂM (YEARLY RETURNS BREAKDOWN)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">📅 Phân Tích Lợi Nhuận Từng Năm (Yearly Performance Breakdown)</div>', unsafe_allow_html=True)
    
    y_comb = m_comb['Yearly Returns']
    y_ema = m_ema['Yearly Returns']
    y_obv = m_obv['Yearly Returns']
    y_bnh = m_bnh['Yearly Returns']

    all_years = sorted(list(set(list(y_comb.keys()) + list(y_bnh.keys()))))
    yearly_table = []
    for yr in all_years:
        r_c = y_comb.get(yr, np.nan)
        r_e = y_ema.get(yr, np.nan)
        r_o = y_obv.get(yr, np.nan)
        r_b = y_bnh.get(yr, np.nan)
        diff_b = (r_c - r_b) if not (np.isnan(r_c) or np.isnan(r_b)) else np.nan
        yearly_table.append({
            'Năm': str(yr),
            '⭐ EMA + OBV (%)': f"{r_c:+.2f}%" if not np.isnan(r_c) else "—",
            'EMA Riêng (%)': f"{r_e:+.2f}%" if not np.isnan(r_e) else "—",
            'OBV Riêng (%)': f"{r_o:+.2f}%" if not np.isnan(r_o) else "—",
            'Benchmark B&H (%)': f"{r_b:+.2f}%" if not np.isnan(r_b) else "—",
            'Chênh Lệch vs B&H (%)': f"{diff_b:+.2f}%" if not np.isnan(diff_b) else "—"
        })

    col_yt_l, col_yt_r = st.columns([5, 5])
    with col_yt_l:
        st.markdown("###### 📊 Bảng Lợi Nhuận Hàng Năm")
        st.dataframe(pd.DataFrame(yearly_table), use_container_width=True, hide_index=True)

    with col_yt_r:
        st.markdown("###### 📊 Biểu Đồ Lợi Nhuận So Sánh Hàng Năm")
        df_y_plot = pd.DataFrame({
            'Năm': [str(yr) for yr in all_years],
            'EMA + OBV': [y_comb.get(yr, 0) for yr in all_years],
            'Benchmark': [y_bnh.get(yr, 0) for yr in all_years]
        })
        fig_y = px.bar(
            df_y_plot, x='Năm', y=['EMA + OBV', 'Benchmark'],
            barmode='group',
            color_discrete_map={'EMA + OBV': '#059669', 'Benchmark': '#94A3B8'},
            labels={'value': 'Lợi Nhuận (%)', 'variable': 'Phương Án'}
        )
        fig_y.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), template="plotly_white")
        st.plotly_chart(fig_y, use_container_width=True)

    # --------------------------------------------------------------------------
    # TOP 5 ĐỢT SỤT GIẢM TÀI SẢN SÂU NHẤT (TOP 5 DRAWDOWN PERIODS)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-header">🌊 Top 5 Đợt Sụt Giảm Tài Sản Sâu Nhất (Top 5 Deepest Drawdowns)</div>', unsafe_allow_html=True)
    st.caption("Chi tiết các đợt sụt giảm lớn nhất của Chiến lược EMA + OBV: Ngày bắt đầu đỉnh, Ngày chạm đáy, Ngày hồi phục và Thời gian chịu đựng.")
    
    top_dd_display = act_comb['top_drawdowns'].copy()
    if len(top_dd_display) > 0:
        top_dd_display['Peak Date'] = pd.to_datetime(top_dd_display['Peak Date']).dt.strftime('%d/%m/%Y')
        top_dd_display['Trough Date'] = pd.to_datetime(top_dd_display['Trough Date']).dt.strftime('%d/%m/%Y')
        top_dd_display['Recovery Date'] = pd.to_datetime(top_dd_display['Recovery Date']).dt.strftime('%d/%m/%Y').fillna('Chưa hồi phục')
        top_dd_display['Max Drawdown (%)'] = top_dd_display['Max Drawdown (%)'].apply(lambda x: f"-{x:.2f}%")
        top_dd_display['Duration (Days)'] = top_dd_display['Duration (Days)'].apply(lambda x: f"{x} ngày")
        top_dd_display.columns = ['Ngày Đỉnh (Peak)', 'Ngày Đáy (Trough)', 'Ngày Phục Hồi (Recovery)', 'Độ Sâu MDD (%)', 'Thời Gian Kéo Dài']
        st.dataframe(top_dd_display, use_container_width=True, hide_index=True)
    else:
        st.info("Không phát hiện đợt sụt giảm nào đáng kể.")

    # --------------------------------------------------------------------------
    # SO SÁNH ĐỐI KHÁNG TRAIN VS TEST
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-header">⚖️ So Sánh Đa Chỉ Số: Tập Huấn Luyện (Train) vs Tập Kiểm Định (Test)</div>', unsafe_allow_html=True)
    st.caption("Trực quan hóa sự suy giảm hiệu suất ngoài mẫu (Out-of-sample deterioration) cho toàn bộ các chỉ số: Sharpe, Calmar, Sortino, Volatility, MDD, Win Rate.")

    if bt_comb_test is not None:
        comp_metrics_data = [
            {'Chiến lược': 'EMA Riêng lẻ', 'Tập': 'Train', 'Sharpe': bt_ema_train['metrics']['Sharpe Ratio'], 'Sortino': bt_ema_train['metrics']['Sortino Ratio'], 'Calmar': bt_ema_train['metrics']['Calmar Ratio'], 'Total Return (%)': bt_ema_train['metrics']['Total Return [%]'], 'Volatility (%)': bt_ema_train['metrics']['Annual Volatility [%]'], 'MDD (%)': -bt_ema_train['metrics']['Max Drawdown [%]'], 'Win Rate (%)': bt_ema_train['metrics']['Winning Rate [%]']},
            {'Chiến lược': 'EMA Riêng lẻ', 'Tập': 'Test', 'Sharpe': bt_ema_test['metrics']['Sharpe Ratio'], 'Sortino': bt_ema_test['metrics']['Sortino Ratio'], 'Calmar': bt_ema_test['metrics']['Calmar Ratio'], 'Total Return (%)': bt_ema_test['metrics']['Total Return [%]'], 'Volatility (%)': bt_ema_test['metrics']['Annual Volatility [%]'], 'MDD (%)': -bt_ema_test['metrics']['Max Drawdown [%]'], 'Win Rate (%)': bt_ema_test['metrics']['Winning Rate [%]']},
            {'Chiến lược': 'OBV Riêng lẻ', 'Tập': 'Train', 'Sharpe': bt_obv_train['metrics']['Sharpe Ratio'], 'Sortino': bt_obv_train['metrics']['Sortino Ratio'], 'Calmar': bt_obv_train['metrics']['Calmar Ratio'], 'Total Return (%)': bt_obv_train['metrics']['Total Return [%]'], 'Volatility (%)': bt_obv_train['metrics']['Annual Volatility [%]'], 'MDD (%)': -bt_obv_train['metrics']['Max Drawdown [%]'], 'Win Rate (%)': bt_obv_train['metrics']['Winning Rate [%]']},
            {'Chiến lược': 'OBV Riêng lẻ', 'Tập': 'Test', 'Sharpe': bt_obv_test['metrics']['Sharpe Ratio'], 'Sortino': bt_obv_test['metrics']['Sortino Ratio'], 'Calmar': bt_obv_test['metrics']['Calmar Ratio'], 'Total Return (%)': bt_obv_test['metrics']['Total Return [%]'], 'Volatility (%)': bt_obv_test['metrics']['Annual Volatility [%]'], 'MDD (%)': -bt_obv_test['metrics']['Max Drawdown [%]'], 'Win Rate (%)': bt_obv_test['metrics']['Winning Rate [%]']},
            {'Chiến lược': 'EMA + OBV Kết hợp', 'Tập': 'Train', 'Sharpe': bt_comb_train['metrics']['Sharpe Ratio'], 'Sortino': bt_comb_train['metrics']['Sortino Ratio'], 'Calmar': bt_comb_train['metrics']['Calmar Ratio'], 'Total Return (%)': bt_comb_train['metrics']['Total Return [%]'], 'Volatility (%)': bt_comb_train['metrics']['Annual Volatility [%]'], 'MDD (%)': -bt_comb_train['metrics']['Max Drawdown [%]'], 'Win Rate (%)': bt_comb_train['metrics']['Winning Rate [%]']},
            {'Chiến lược': 'EMA + OBV Kết hợp', 'Tập': 'Test', 'Sharpe': bt_comb_test['metrics']['Sharpe Ratio'], 'Sortino': bt_comb_test['metrics']['Sortino Ratio'], 'Calmar': bt_comb_test['metrics']['Calmar Ratio'], 'Total Return (%)': bt_comb_test['metrics']['Total Return [%]'], 'Volatility (%)': bt_comb_test['metrics']['Annual Volatility [%]'], 'MDD (%)': -bt_comb_test['metrics']['Max Drawdown [%]'], 'Win Rate (%)': bt_comb_test['metrics']['Winning Rate [%]']},
        ]
        comp_metrics_df = pd.DataFrame(comp_metrics_data)

        row_c1, row_c2, row_c3 = st.columns(3)
        with row_c1:
            fig_c1 = px.bar(comp_metrics_df, x='Chiến lược', y='Calmar', color='Tập', barmode='group',
                            title='Calmar Ratio (Train vs Test)', text_auto='.2f',
                            color_discrete_map={'Train': '#059669', 'Test': '#D97706'})
            fig_c1.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_c1, use_container_width=True)

        with row_c2:
            fig_c2 = px.bar(comp_metrics_df, x='Chiến lược', y='Sortino', color='Tập', barmode='group',
                            title='Sortino Ratio (Train vs Test)', text_auto='.2f',
                            color_discrete_map={'Train': '#2563EB', 'Test': '#EC4899'})
            fig_c2.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_c2, use_container_width=True)

        with row_c3:
            fig_c3 = px.bar(comp_metrics_df, x='Chiến lược', y='Win Rate (%)', color='Tập', barmode='group',
                            title='Winning Rate (%) (Train vs Test)', text_auto='.1f',
                            color_discrete_map={'Train': '#10B981', 'Test': '#64748B'})
            fig_c3.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_c3, use_container_width=True)

# ==============================================================================
# TAB 3: QUẢN LÝ VỊ THẾ & NHẬT KÝ LỆNH
# ==============================================================================
with tab3:
    st.subheader("Hạch Toán Vị Thế (Position Accounting) & Nhật Ký Giao Dịch")
    
    st.markdown("""
    > **Trọng tâm sửa đổi theo góp ý chuyên gia:** Chiến lược chỉ duy trì vị thế Long (`direction='longonly'`),
    > không nhồi lệnh (`accumulate=False`). Một lệnh mới chỉ được mở khi tài khoản đang ở trạng thái tiền mặt (`position == 0`).
    """)

    trade_set = st.selectbox("Chọn tập dữ liệu xem nhật ký giao dịch:", ["Tập Train (2014 - 2020)", "Tập Test (2021 - 2023)", "Toàn Bộ (2014 - 2023)"], index=0)
    if "Train" in trade_set:
        active_trades_df = bt_comb_train['trades_df']
        active_pos_series = bt_comb_train['positions']
    elif "Test" in trade_set:
        active_trades_df = bt_comb_test['trades_df'] if bt_comb_test else pd.DataFrame()
        active_pos_series = bt_comb_test['positions'] if bt_comb_test else pd.Series()
    else:
        active_trades_df = bt_comb_full['trades_df']
        active_pos_series = bt_comb_full['positions']

    # 1. Timeline Vị Thế
    st.markdown("##### ⏱️ Dòng Thời Gian Trạng Thái Vị Thế (0: Tiền mặt | 1: Cổ phiếu)")
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(
        x=active_pos_series.index, y=active_pos_series,
        mode='lines', fill='tozeroy', name='Trạng Thái Vị Thế',
        line=dict(color='#2563EB', width=1.5),
        fillcolor='rgba(37, 99, 235, 0.25)'
    ))
    fig_pos.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=20, b=20),
        yaxis=dict(tickvals=[0, 1], ticktext=["0: Tiền Mặt", "1: Đang Giữ Hàng"]),
        template="plotly_white"
    )
    st.plotly_chart(fig_pos, use_container_width=True)

    # 2. Nhật ký giao dịch
    st.markdown("##### 📜 Danh Sách Chi Tiết Từng Lệnh Giao Dịch Đã Đóng")
    if len(active_trades_df) > 0:
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            st.caption(f"Tổng số lệnh đã đóng: **{len(active_trades_df)}** | Số lệnh thắng: **{len(active_trades_df[active_trades_df['PnL (%)'] > 0])}** | Số lệnh lỗ: **{len(active_trades_df[active_trades_df['PnL (%)'] <= 0])}**")
        with col_t2:
            csv_buffer = io.StringIO()
            active_trades_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Tải Nhật Ký Lệnh (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"trade_log_{trade_set.replace(' ', '_')}.csv",
                mime="text/csv"
            )

        display_trade_df = active_trades_df.copy()
        display_trade_df['Entry Date'] = pd.to_datetime(display_trade_df['Entry Date']).dt.strftime('%d/%m/%Y')
        display_trade_df['Exit Date'] = pd.to_datetime(display_trade_df['Exit Date']).dt.strftime('%d/%m/%Y')
        display_trade_df['Entry Price'] = display_trade_df['Entry Price'].apply(lambda x: f"{x:,.0f}")
        display_trade_df['Exit Price'] = display_trade_df['Exit Price'].apply(lambda x: f"{x:,.0f}")
        display_trade_df['PnL (VND)'] = display_trade_df['PnL (VND)'].apply(lambda x: f"{x:+,.0f}")
        display_trade_df['PnL (%)'] = display_trade_df['PnL (%)'].apply(lambda x: f"{x:+.2f}%")

        st.dataframe(display_trade_df, use_container_width=True, hide_index=True)

        st.markdown("##### 📊 Phân Phối Tỷ Suất Sinh Lời Từng Lệnh (Trade Return Distribution)")
        fig_hist = px.histogram(
            active_trades_df, x="PnL (%)", nbins=25,
            color=active_trades_df['PnL (%)'] > 0,
            color_discrete_map={True: '#059669', False: '#DC2626'},
            labels={'color': 'Lệnh Lời'},
            title="Biểu đồ tần suất lợi nhuận các lệnh (%)"
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#475569")
        fig_hist.update_layout(template="plotly_white", height=320, showlegend=False, margin=dict(l=20, r=20, t=35, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Chưa có giao dịch nào được thực hiện trong khoảng thời gian này.")

# ==============================================================================
# TAB 4: TỐI ƯU HÓA THAM SỐ (HYPEROPT TPE OPTIMIZATION)
# ==============================================================================
with tab4:
    st.subheader("Tối Ưu Hóa Tham Số Bằng Thuật Toán TPE (Hyperopt)")
    
    st.markdown("""
    **Cơ chế giải quyết lỗi 'Sharpe ảo do ít giao dịch':**
    Một bộ tham số chỉ được xem là hợp lệ khi tạo ra ít nhất `MIN_TRADES` giao dịch.
    Nếu số lệnh ít hơn ngưỡng này, hàm mục tiêu sẽ phạt điểm nặng (`loss = inf`), ngăn thuật toán
    chọn các bộ tham số 'đứng ngoài thị trường' ăn may.
    """)

    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        n_evals = st.selectbox("Số lượt đánh giá (Max Evaluations):", [30, 50, 100, 200], index=1)
    with col_opt2:
        ema_search_range = st.slider("Phạm vi quét chu kỳ EMA:", min_value=10, max_value=60, value=(10, 50), step=2)
    with col_opt3:
        obv_search_range = st.slider("Phạm vi quét chu kỳ Độ dốc OBV:", min_value=2, max_value=25, value=(2, 20), step=1)

    if st.button("🚀 Bắt Đầu Quét Tối Ưu Hóa Trên Tập Train", type="primary"):
        if not HAS_HYPEROPT:
            st.error("Thư viện hyperopt chưa được cài đặt. Vui lòng kiểm tra lại requirements.txt.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Đang khởi tạo không gian tham số và chạy thuật toán TPE...")

            search_space = {
                'ema_period': hp.quniform('ema_period', ema_search_range[0], ema_search_range[1], 1),
                'obv_slope_period': hp.quniform('obv_slope_period', obv_search_range[0], obv_search_range[1], 1)
            }

            eval_counter = [0]
            trial_results = []

            def objective_func(params):
                eval_counter[0] += 1
                progress = min(eval_counter[0] / n_evals, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Đang đánh giá bộ tham số {eval_counter[0]}/{n_evals}...")

                ema_p = int(params['ema_period'])
                obv_p = int(params['obv_slope_period'])

                df_temp = calculate_indicators(train_df, ema_p, obv_p)
                sigs = generate_all_signals(df_temp, ema_p, obv_p)
                bt = run_backtest_engine(df_temp, sigs['Combined'][0], sigs['Combined'][1],
                                         initial_capital, fee_pct, slippage_pct, stop_loss_pct)

                n_trades = bt['metrics']['Total Closed Trades']
                sharpe = bt['metrics']['Sharpe Ratio']
                calmar = bt['metrics']['Calmar Ratio']
                total_ret = bt['metrics']['Total Return [%]']
                mdd = bt['metrics']['Max Drawdown [%]']

                valid = np.isfinite(sharpe) and (n_trades >= min_trades_filter)
                loss = -sharpe if valid else 1e9

                res = {
                    'loss': loss,
                    'status': STATUS_OK,
                    'ema_period': ema_p,
                    'obv_slope_period': obv_p,
                    'sharpe': sharpe if np.isfinite(sharpe) else None,
                    'calmar': calmar if np.isfinite(calmar) else None,
                    'total_return': total_ret,
                    'mdd': mdd,
                    'trades': n_trades,
                    'valid': bool(valid)
                }
                trial_results.append(res)
                return res

            trials = Trials()
            best = fmin(
                fn=objective_func,
                space=search_space,
                algo=tpe.suggest,
                max_evals=n_evals,
                trials=trials,
                rstate=np.random.default_rng(42)
            )

            progress_bar.progress(1.0)
            status_text.success("✅ Hoàn tất tối ưu hóa!")

            best_ema_found = int(best['ema_period'])
            best_obv_found = int(best['obv_slope_period'])

            st.balloons()
            st.markdown(f"""
            <div class="signal-box-buy">
                <div style="font-size: 1.3rem; font-weight: 700; color: #065F46;">
                    🎉 BỘ THAM SỐ TỐI ƯU TÌM ĐƯỢC: EMA = {best_ema_found}, OBV Slope = {best_obv_found}
                </div>
                <div style="margin-top: 6px; font-size: 0.95rem; color: #047857;">
                    Bộ tham số này mang lại Sharpe & Calmar Ratio cao nhất trên tập Train và thỏa mãn điều kiện tối thiểu {min_trades_filter} giao dịch.
                </div>
            </div>
            """, unsafe_allow_html=True)

            res_df = pd.DataFrame(trial_results)
            valid_df = res_df[res_df['valid']].sort_values(by='loss').reset_index(drop=True)

            st.markdown("##### 🏆 Top 5 Bộ Tham Số Xuất Sắc Nhất")
            top_display = valid_df.head(5)[['ema_period', 'obv_slope_period', 'sharpe', 'calmar', 'total_return', 'mdd', 'trades']].copy()
            top_display.columns = ['EMA Period', 'OBV Slope', 'Sharpe Ratio', 'Calmar Ratio', 'Tổng LN (%)', 'MDD (%)', 'Số Lệnh Đã Đóng']
            top_display['Sharpe Ratio'] = top_display['Sharpe Ratio'].apply(lambda x: f"{x:.4f}")
            top_display['Calmar Ratio'] = top_display['Calmar Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "—")
            top_display['Tổng LN (%)'] = top_display['Tổng LN (%)'].apply(lambda x: f"{x:+.2f}%")
            top_display['MDD (%)'] = top_display['MDD (%)'].apply(lambda x: f"-{x:.2f}%")
            st.dataframe(top_display, use_container_width=True, hide_index=True)

            st.markdown("##### 🌌 Biểu Đồ Không Gian Tham Số (Parameter Space vs Sharpe)")
            fig_space = px.scatter(
                valid_df, x='ema_period', y='obv_slope_period', size=np.maximum(valid_df['trades'], 1),
                color='sharpe', color_continuous_scale='Viridis',
                title="Tương quan giữa Chu kỳ EMA, Độ dốc OBV và Sharpe Ratio",
                labels={'ema_period': 'Chu kỳ EMA', 'obv_slope_period': 'Chu kỳ Độ dốc OBV', 'sharpe': 'Sharpe Ratio'}
            )
            fig_space.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_space, use_container_width=True)

# ==============================================================================
# TAB 5: PHƯƠNG PHÁP LUẬN & ĐỊNH NGHĨA CHỈ SỐ ĐỊNH LƯỢNG
# ==============================================================================
with tab5:
    st.subheader("Khung Lý Thuyết & Định Nghĩa Các Chỉ Số Định Lượng")

    st.markdown("""
    ### 1. Ý Nghĩa & Công Thức Các Chỉ Số Đo Lường Hiệu Quả

    #### 🟢 Calmar Ratio (Tỷ Số Calmar)
    * **Định nghĩa:** Là tỷ số đo lường tỷ suất sinh lời kép hàng năm (**CAGR**) so với mức sụt giảm tối đa (**Maximum Drawdown - MDD**) trong suốt thời kỳ kiểm định:
      $$\\text{Calmar Ratio} = \\frac{\\text{CAGR [\\%]}}{|\\text{Max Drawdown [\\%]}|}$$
    * **Ý nghĩa thực tiễn:** Đo lường mỗi 1% rủi ro sụt giảm tài sản đổi lại được bao nhiêu % lợi nhuận hàng năm. Tỷ số Calmar rất được các quỹ theo xu hướng (Trend-Following CTA) ưa chuộng:
      * **Calmar > 1.0:** Rất tốt (lợi nhuận hàng năm bù đắp hoàn toàn đợt sập tồi tệ nhất).
      * **Calmar > 2.0:** Xuất sắc.

    ---

    #### 🔵 Sortino Ratio (Tỷ Số Sortino)
    * **Định nghĩa:** Tương tự Sharpe Ratio nhưng **chỉ phạt độ biến động tiêu cực (Downside Deviation)** thay vì phạt cả những đợt tăng giá mạnh:
      $$\\text{Sortino Ratio} = \\frac{\\bar{R} - R_f}{\\sigma_{\\text{downside}}}$$
    * **Ưu điểm so với Sharpe:** Trong chiến lược theo sau xu hướng, lợi nhuận thường có độ lệch dương lớn (những phiên trần, bứt phá mạnh). Sharpe Ratio sẽ xem đà tăng mạnh này là "biến động rủi ro" và phạt điểm, trong khi Sortino Ratio chỉ tính rủi ro khi danh mục bị giảm giá.

    ---

    #### 📉 Maximum Drawdown (MDD - Mức Sụt Giảm Tối Đa)
    * **Định nghĩa:** Là tỷ lệ sụt giảm sâu nhất của tài sản từ một đỉnh cao lịch sử (Peak) xuống đáy thấp nhất (Trough) trước khi lập đỉnh mới:
      $$\\text{Drawdown}_t = \\frac{\\text{Equity}_t - \\max_{0 \\le s \\le t}(\\text{Equity}_s)}{\\max_{0 \\le s \\le t}(\\text{Equity}_s)}$$
      $$\\text{MDD} = \\min_t (\\text{Drawdown}_t)$$
    * **Ý nghĩa:** Cho biết mức thua lỗ tồi tệ nhất mà nhà đầu tư phải chịu đựng nếu không may tham gia đúng ngay đỉnh thị trường.

    ---

    #### 🏆 Winning Rate & Payoff Ratio
    * **Tỷ Lệ Thắng (Winning Rate):** $\\frac{\\text{Số lệnh thắng}}{\\text{Tổng số lệnh}} \\times 100\\%$.
    * **Payoff Ratio (Tỷ số Thắng/Thua):** $\\frac{\\text{Lãi bình quân lệnh thắng}}{\\text{Lỗ bình quân lệnh thua}}$.
    * **Bản chất chiến lược Trend Following:** Chiến lược theo xu hướng thường có tỷ lệ thắng không cao (chỉ khoảng **35% – 45%**), nhưng có **Payoff Ratio rất cao (2.5x – 4.0x)**: "Cắt lỗ thật nhanh khi sai (chỉ mất 2% – 7%) và gồng lãi thật dài khi đúng (lãi 20% – 40%)".

    ---

    ### 2. Các Đóng Góp Học Thuật Sửa Lỗi Trong Nghiên Cứu
    1. **Khắc phục Look-ahead Bias:** Trễ 1 phiên (`shift(1)`) trên toàn bộ chuỗi tín hiệu Mua/Bán.
    2. **Hạch toán Vị thế (Position Accounting):** Mô hình *Long-only*, chỉ mở tối đa 1 vị thế (`accumulate=False`), không bán khống khi giữ tiền mặt.
    3. **Loại bỏ Bẫy Sharpe Vô Cực:** Ràng buộc `MIN_TRADES >= 5` để loại trừ các bộ tham số 'đứng ngoài thị trường' không vào lệnh.
    4. **Quản trị Rủi ro:** Stop loss cứng 7% và đưa chi phí giao dịch thực tế (phí 0.2% + trượt giá 0.1%).
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94A3B8; font-size: 0.85rem;'>"
    "Hệ thống Kiểm định Chiến lược Định lượng EMA & OBV | Được xây dựng cho nền tảng Streamlit Cloud & GitHub | © 2024"
    "</div>",
    unsafe_allow_html=True
)
