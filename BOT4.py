import ccxt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import time
import threading
from datetime import datetime
from tabulate import tabulate
from colorama import Fore, Style, init
import concurrent.futures
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import sys

# Khởi tạo màu console
init(autoreset=True)

# ================= CẤU HÌNH HỆ THỐNG (FINAL) =================
SYSTEM_CONFIG = {
    # --- CẤU HÌNH SÀN ---
    'exchange': 'binance',
    'timeframe': '15m',
    'limit_history': 1000,      # Số nến để scan
    
    # --- CẤU HÌNH SCANNER ---
    'scan_interval': 300,       # Quét lại sau mỗi 5 phút (300s)
    'p_value_threshold': 0.049,   # Ngưỡng P-value
    
    # --- CẤU HÌNH TRADING BOT ---
    'kf_delta': 1e-4,           # Kalman Filter Delta
    'kf_vt': 1e-3,              # Kalman Filter Vt
    'entry_z': 2.0,             # Ngưỡng vào lệnh Z-Score
    'exit_z': 0.0,              # Ngưỡng thoát lệnh
    'stop_loss_z': 4.5,

    'auto_optimize_z': True,    # Bật tính năng tự tìm Z tối ưu
    'min_entry_z': 1.5,         # Không bao giờ vào lệnh thấp hơn 1.5 (để đảm bảo lợi nhuận)
    'max_entry_z': 3.0,         # Không bao giờ đợi quá 3.0 (để tránh miss kèo)
    'z_percentile': 90,         # Chọn ngưỡng mà 90% các đỉnh trong quá khứ đều chạm tới

    'min_profit_pct': 0.003,    # 0.3%
    'fixed_loss_usdt': 5,     # số USDT chấp nhận mất cố định cho mỗi lệnh
    'max_loss_usdt': 10.0,     # Mức lỗ tối đa chấp nhận cho mỗi lệnh (USDT)
    'leverage': 50,
    # --- CẤU HÌNH TIME STOP (MỚI) ---
    'time_stop_factor': 2.0,    # Thoát lệnh nếu giữ quá 2.0 lần Half-Life

    # --- TỐI ƯU HÓA TỐC ĐỘ ---
    'bot_scan_interval': 60,    # 60 Giây (1 Phút) Bot check giá 1 lần
    'show_heartbeat': True,     # HIỆN LOG KHI BOT ĐANG CHỜ
    
    # --- CẤU HÌNH EMAIL ---
    'email_enabled': True,
    'email_sender': 'vuongtinhkhac@gmail.com',
    'email_password': 'eiow sbkd isqr rtnu', 
    'email_receiver': 'vuongtinhkhac@gmail.com',
    'email_cooldown': 3600,     

    # --- CẤU HÌNH SÀN GIAO DỊCH (BINGX) ---
    'bingx_api_key': '6xfkNHeeto3IgZ4IcN9oIcMGxte0hmt2mWAb4tTnuuEqnkwGymQzV4KX6jEULn6sHiUeMGIK5iUUPZUUrcw',
    'bingx_secret_key': 'Qa9mcqlj3ashYlRHeGLCC2Isaq8fj6CSyBIG3GLnk9GWQis9OHdKuAbTl8ewSow0k1wNn3y3I30N7yIJaUg',
    
    # --- DANH SÁCH CẶP SCAN ---
    'pairs_pool': [
        # Nhóm Coin giá trị tương đương (Mid-cap)
        ('DOT/USDT', 'ATOM/USDT'),    # DOT (~8$) > ATOM (~6$)
        ('DOGE/USDT', 'SHIB/USDT'),   # DOGE (0.4$) > SHIB (0.00003$)
        ('BCH/USDT', 'LTC/USDT'),     # BCH (~450$) > LTC (~110$)  <-- Đảo lại
        ('SOL/USDT', 'JUP/USDT'),     # SOL (~235$) > JUP (~1.3$)
        ('OP/USDT', 'ARB/USDT'),      # OP (~2.2$) > ARB (~0.9$)   <-- Đảo lại
        ('LINK/USDT', 'UNI/USDT'),    # LINK (~22$) > UNI (~14$)
        ('XRP/USDT', 'ADA/USDT'),     # XRP (~2.5$) > ADA (~1.2$)  <-- Đảo lại
        ('AVAX/USDT', 'POL/USDT'),    # AVAX (~50$) > POL (~0.6$)  <-- Đảo lại
        ('FTM/USDT', 'POL/USDT'),     # FTM (~1.0$) > POL (~0.6$)  <-- Đảo lại
        ('XLM/USDT', 'ALGO/USDT'),    # XLM (~0.5$) > ALGO (~0.4$)
        ('UNI/USDT', 'SUSHI/USDT'),   # UNI (~14$) > SUSHI (~1.5$)

        # Nhóm ETH làm trụ (ETH giá ~3900$, luôn nằm trước các Altcoin)
        ('ETH/USDT', 'BNB/USDT'),     # ETH > BNB (~700$)          <-- Đảo lại
        ('ETH/USDT', 'ETC/USDT'),     # ETH > ETC                  <-- Đảo lại
        ('ETH/USDT', 'POL/USDT'),     # ETH > POL                  <-- Đảo lại
        ('ETH/USDT', 'AVAX/USDT'),    # ETH > AVAX                 <-- Đảo lại
        ('ETH/USDT', 'SOL/USDT'),     # ETH > SOL                  <-- Đảo lại
        ('ETH/USDT', 'DOT/USDT'),     # ETH > DOT                  <-- Đảo lại
        ('ETH/USDT', 'LINK/USDT'),    # ETH > LINK                 <-- Đảo lại
        ('ETH/USDT', 'UNI/USDT'),     # ETH > UNI                  <-- Đảo lại
        ('ETH/USDT', 'ADA/USDT'),     # ETH > ADA                  <-- Đảo lại
        ('ETH/USDT', 'DOGE/USDT'),    # ETH > DOGE                 <-- Đảo lại
        ('ETH/USDT', 'LTC/USDT'),     # ETH > LTC                  <-- Đảo lại
        ('ETH/USDT', 'BCH/USDT'),     # ETH > BCH                  <-- Đảo lại 

        # Nhóm BTC làm trụ (BTC giá ~98000$, luôn nằm trước tất cả)
        ('BTC/USDT', 'ETH/USDT'),     # BTC > ETH
        ('BTC/USDT', 'BNB/USDT'),     # BTC > BNB                  <-- Đảo lại
        ('BTC/USDT', 'ETC/USDT'),     # BTC > ETC                  <-- Đảo lại
        
        # Sửa cặp XRP/LTC
        ('LTC/USDT', 'XRP/USDT'),     # LTC (~110$) > XRP (~2.5$)  <-- Đảo lại
    ]
}

# ================= MODULE 1: KALMAN FILTER =================
class KalmanFilterReg:
    def __init__(self, delta=1e-4, vt=1e-3):
        self.delta = delta 
        self.vt = vt        
        self.x = np.zeros((2, 1)) # State [beta, alpha]
        self.P = np.zeros((2, 2)) 
        self.R = self.vt           
        self.Q = self.delta / (1 - self.delta) * np.eye(2) 

    def update(self, price_y, price_x):
        H = np.array([[price_x, 1.0]])
        x_pred = self.x 
        P_pred = self.P + self.Q
        y_pred = np.dot(H, x_pred)
        error = price_y - y_pred 
        S = np.dot(np.dot(H, P_pred), H.T) + self.R 
        K = np.dot(P_pred, H.T) / S                  
        self.x = x_pred + K * error
        self.P = P_pred - np.dot(np.dot(K, H), P_pred)
        return self.x[0, 0], self.x[1, 0], error[0, 0]

# ================= MODULE 2: TRADING BOT WORKER =================
class TradingBotWorker(threading.Thread):
    def __init__(self, symbol_y, symbol_x, z_window, initial_history, initial_hl):
        super().__init__()
        self.symbol_y = symbol_y
        self.symbol_x = symbol_x
        self.z_window = int(z_window)
        self.pair_name = f"{symbol_y}-{symbol_x}"
        
        # Khởi tạo exchange riêng
        self.exchange_data = getattr(ccxt, SYSTEM_CONFIG['exchange'])({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'} 
        })

        self.exchange_exec = ccxt.bingx({
            'apiKey': SYSTEM_CONFIG['bingx_api_key'],
            'secret': SYSTEM_CONFIG['bingx_secret_key'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap', 
                'adjustForTimeDifference': True
            }
        })

        try:
            self.exchange_exec.load_markets()
            #self.set_leverage(SYSTEM_CONFIG['leverage'])
        except Exception as e:
            print(f"Lỗi tải market BingX: {e}")
        
        self.kf = KalmanFilterReg(delta=SYSTEM_CONFIG['kf_delta'], vt=SYSTEM_CONFIG['kf_vt'])
        self.running = True
        
        # Bộ nhớ thống kê
        self.spread_history = []
        self.cached_mean = 0
        self.cached_std = 0
        self.cached_beta = 0 
        self.cached_alpha = 0
        
        # Biến quản lý trạng thái rủi ro
        self.current_position_state = 'NEUTRAL'
        self.latest_p_value = 0.0 
        self.force_exit_trigger = False
        
        self.last_processed_candle_ts = None

        self.entry_time = None  # Thời điểm vào lệnh
        self.latest_half_life = initial_hl # Half-life cập nhật liên tục
        
        # Các biến lưu giá và số lượng để tính PnL USDT
        self.entry_price_y = 0.0
        self.entry_price_x = 0.0
        self.qty_y = 0.0
        self.qty_x = 0.0

        self.dynamic_entry_z = SYSTEM_CONFIG['entry_z'] # Mặc định
        # Nạp dữ liệu ban đầu
        self.init_warmup(initial_history)

    def set_leverage(self, leverage):
        try:
            # Cài đòn bẩy cho cả 2 cặp
            symbol_y_f = self.get_bingx_futures_symbol(self.symbol_y)
            symbol_x_f = self.get_bingx_futures_symbol(self.symbol_x)
            self.exchange_exec.set_leverage(leverage, symbol_y_f)
            self.exchange_exec.set_leverage(leverage, symbol_x_f)
            print(f"✅ Đã cài đòn bẩy {leverage}x cho {self.symbol_y} & {self.symbol_x}")
        except: pass # Có thể lỗi nếu đã cài rồi, bỏ qua

    def log(self, msg, color=Fore.WHITE):
        """Hàm log riêng để in tên Bot kèm theo"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{Fore.CYAN}[BOT {self.pair_name}]{Style.RESET_ALL} {timestamp} | {color}{msg}{Style.RESET_ALL}")

    def update_p_value_and_halflife(self, p_val, half_life):
        """Hàm này được Scanner gọi để cập nhật P-value mới nhất cho Bot"""
        self.latest_p_value = p_val
        self.latest_half_life = half_life
        # Nếu đang giữ lệnh mà P-value xấu, in cảnh báo ngay
        if self.current_position_state != 'NEUTRAL' and p_val > SYSTEM_CONFIG['p_value_threshold']:
            self.log(f"⚠️ CẢNH BÁO: P-Value tăng cao ({p_val:.4f}). Chuẩn bị thoát lệnh!", Fore.RED)

    def init_warmup(self, df_merged):
        try:
            self.kf = KalmanFilterReg(delta=SYSTEM_CONFIG['kf_delta'], vt=SYSTEM_CONFIG['kf_vt'])
            self.spread_history = []
            z_history = [] # Lưu lịch sử Z-score để phân tích
            
            last_beta = 0
            last_alpha = 0
            
            for idx, row in df_merged.iterrows():
                beta, alpha, spread = self.kf.update(row['close_y'], row['close_x'])
                self.spread_history.append(spread)
                last_beta = beta
                last_alpha = alpha

                # Tính Z-Score lịch sử (cần ít nhất z_window nến)
                if len(self.spread_history) >= self.z_window:
                    window_spread = self.spread_history[-self.z_window:]
                    mean = np.mean(window_spread)
                    std = np.std(window_spread)
                    if std != 0:
                        z = (spread - mean) / std
                        z_history.append(abs(z)) # Lưu giá trị tuyệt đối |Z|
            
            if len(self.spread_history) > self.z_window:
                self.spread_history = self.spread_history[-self.z_window:]
            
            series = pd.Series(self.spread_history)
            self.cached_mean = series.mean()
            self.cached_std = series.std()
            self.cached_beta = last_beta
            self.cached_alpha = last_alpha
            
            last_ts = pd.to_datetime(df_merged.index[-1])
            self.last_processed_candle_ts = (last_ts.hour * 60 + last_ts.minute) // 15

            # =================================================================
            #  TỰ ĐỘNG TÌM NGƯỠNG ENTRY Z TỐI ƯU (AUTO OPTIMIZER)
            # =================================================================
            if SYSTEM_CONFIG['auto_optimize_z'] and len(z_history) > 100:
                # Chuyển thành Numpy array để tính toán
                z_arr = np.array(z_history)
                
                # Lọc ra các đỉnh (Peaks) quan trọng: Những điểm Z > 1.0 (Bỏ qua nhiễu nhỏ)
                # Tư duy: Chúng ta chỉ quan tâm những lần giá đi xa trung bình
                significant_moves = z_arr[z_arr > 1.0]
                
                if len(significant_moves) > 10:
                    # Tính Phân vị thứ 90 (90th Percentile)
                    # Ý nghĩa: 90% các cú biến động mạnh trong quá khứ đều nằm dưới mức này.
                    # -> Nếu đặt Entry ở đây, ta bắt được các cú swing mạnh nhất.
                    optimal_z = np.percentile(significant_moves, SYSTEM_CONFIG['z_percentile'])
                    
                    # Kẹp giá trị trong khoảng an toàn (Min/Max Cap)
                    # Ví dụ: Tính ra 4.0 thì ép xuống 3.0. Tính ra 1.2 thì ép lên 1.5
                    optimal_z = max(SYSTEM_CONFIG['min_entry_z'], min(optimal_z, SYSTEM_CONFIG['max_entry_z']))
                    
                    self.dynamic_entry_z = optimal_z
                    
                    self.log(f"🧠 AI OPTIMIZED Z: {self.dynamic_entry_z:.2f} (Based on {SYSTEM_CONFIG['z_percentile']}th Percentile)", Fore.YELLOW)
                else:
                    self.log(f"⚠️ Not enough volatility for optimization. Using default Z: {self.dynamic_entry_z}", Fore.LIGHTBLACK_EX)
            
            self.log(f"INIT SUCCESS | Entry Z: {self.dynamic_entry_z:.2f} | Beta: {last_beta:.4f}", Fore.GREEN)
        except Exception as e:
            self.log(f"INIT ERROR: {e}", Fore.RED)
            self.stop()

    def re_calibrate(self):
        self.log("Đang Re-calibrate (tải lại dữ liệu)...", Fore.CYAN)
        try:
            limit = max(500, self.z_window * 2)
            ohlcv_y = self.exchange_data.fetch_ohlcv(self.symbol_y, SYSTEM_CONFIG['timeframe'], limit=limit)
            ohlcv_x = self.exchange_data.fetch_ohlcv(self.symbol_x, SYSTEM_CONFIG['timeframe'], limit=limit)
            
            df_y = pd.DataFrame(ohlcv_y, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            df_x = pd.DataFrame(ohlcv_x, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            df_y['ts'] = pd.to_datetime(df_y['ts'], unit='ms')
            df_x['ts'] = pd.to_datetime(df_x['ts'], unit='ms')
            
            df = pd.merge(df_y[['ts', 'c']], df_x[['ts', 'c']], on='ts', suffixes=('_y', '_x'))
            df.columns = ['timestamp', 'close_y', 'close_x']
            df.set_index('timestamp', inplace=True)
            
            self.init_warmup(df)
            
        except Exception as e:
            self.log(f"Re-calibrate Failed: {e}", Fore.RED)

    def fetch_current_price(self):
        try:
            ticker_y = self.exchange_data.fetch_ticker(self.symbol_y)
            ticker_x = self.exchange_data.fetch_ticker(self.symbol_x)
            return ticker_y['last'], ticker_x['last']
        except:
            return None, None

    def get_bingx_futures_symbol(self, symbol):
        # BingX Futures thường cần định dạng "ETH/USDT:USDT" để phân biệt với Spot
        # Hàm này sẽ tìm trong market đã load để lấy ID chính xác
        try:
            market = self.exchange_exec.market(symbol)
            return market['symbol'] # CCXT thường tự map sang 'ETH/USDT:USDT' nếu defaultType=swap
        except:
            # Nếu không tìm thấy, ép kiểu thủ công
            if ':' not in symbol:
                return f"{symbol}:USDT"
            return symbol

    # HÀM CHUẨN HÓA SỐ LƯỢNG THEO QUY TẮC BINGX
    def normalize_amount(self, symbol, amount):
        try:
            # Dùng symbol chuẩn Futures để normalize
            bingx_symbol = self.get_bingx_futures_symbol(symbol)
            return float(self.exchange_exec.amount_to_precision(bingx_symbol, amount))
        except Exception as e:
            self.log(f"⚠ Lỗi Normalize {symbol}: {e}", Fore.YELLOW)
            return amount

    def execute_bingx_order(self, symbol, side, amount):
        """
        side: 'buy' hoặc 'sell'
        amount: số lượng coin (Quantity)
        """
        try:
            target_symbol = self.get_bingx_futures_symbol(symbol)
            params = {}
            if side == 'buy':
                params['positionSide'] = 'LONG'  # Mua là mở Long
            elif side == 'sell':
                params['positionSide'] = 'SHORT' # Bán là mở Short
            order = self.exchange_exec.create_order(target_symbol, 'market', side, amount, params=params)
            self.log(f"✅ BINGX ORDER: {side.upper()} {amount} {symbol}", Fore.GREEN)
            return order
        except Exception as e:
            self.log(f"❌ BINGX ORDER ERROR ({symbol}): {e}", Fore.RED)
            return None 
    
    def execute_bingx_close(self, symbol, side, amount):
        """
        side: 'buy' hoặc 'sell'
        amount: số lượng coin (Quantity)
        """
        try:
            target_symbol = self.get_bingx_futures_symbol(symbol)
            params = {}
            if side == 'buy':
                params['positionSide'] = 'SHORT' 
            elif side == 'sell':
                params['positionSide'] = 'LONG' 
            order = self.exchange_exec.create_order(target_symbol, 'market', side, amount, params=params)
            self.log(f"✅ BINGX CLOSE: {side.upper()} {amount} {symbol}", Fore.GREEN)
            return order
        except Exception as e:
            self.log(f"❌ BINGX CLOSE ERROR ({symbol}): {e}", Fore.RED)
            return None 

    def send_email(self, subject, content):
        if not SYSTEM_CONFIG['email_enabled']: return
        try:
            msg = MIMEMultipart()
            msg['From'] = SYSTEM_CONFIG['email_sender']
            msg['To'] = SYSTEM_CONFIG['email_receiver']
            msg['Subject'] = subject
            msg.attach(MIMEText(content, 'html'))
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(SYSTEM_CONFIG['email_sender'], SYSTEM_CONFIG['email_password'])
                server.send_message(msg)
        except Exception as e:
            print(f"Mail Error: {e}")

    def run(self):
        while self.running:
            try:
                now = datetime.now()
                current_candle_ts = (now.hour * 60 + now.minute) // 15
                
                py, px = self.fetch_current_price()
                
                if py and px:
                    # 1. Update Thống kê (Mỗi khi đóng nến 15m)
                    if self.last_processed_candle_ts is not None and current_candle_ts != self.last_processed_candle_ts:
                        if self.current_position_state == 'NEUTRAL':
                            beta_new, alpha_new, spread_new = self.kf.update(py, px)
                            self.spread_history.append(spread_new)
                            if len(self.spread_history) > self.z_window: self.spread_history.pop(0)
                            
                            series = pd.Series(self.spread_history)
                            self.cached_mean = series.mean()
                            self.cached_std = series.std()
                            self.cached_beta = beta_new
                            self.cached_alpha = alpha_new
                            
                            self.log(f"Update Stats M15. Beta: {beta_new:.4f}", Fore.MAGENTA)
                        
                        self.last_processed_candle_ts = current_candle_ts

                    # 2. Tính toán Realtime
                    calc_beta = self.cached_beta
                    calc_alpha = self.cached_alpha
                    live_spread = py - (calc_beta * px + calc_alpha)
                    
                    if self.cached_std == 0: z_score = 0
                    else: z_score = (live_spread - self.cached_mean) / self.cached_std
                    
                    # [TRỌNG TÂM] TÍNH TOÁN PnL BÙ TRỪ (NET PNL) RA SỐ USDT
                    net_pnl_usdt = 0.0
                    
                    if self.current_position_state != 'NEUTRAL':
                        # Tính PnL cho chân Y
                        pnl_y = 0
                        if self.current_position_state == 'LONG': # Đang Long Y
                            pnl_y = (py - self.entry_price_y) * self.qty_y
                        else: # Đang Short Y
                            pnl_y = (self.entry_price_y - py) * self.qty_y
                            
                        # Tính PnL cho chân X
                        pnl_x = 0
                        if self.current_position_state == 'LONG': # Long Spread = Short X
                            pnl_x = (self.entry_price_x - px) * self.qty_x # Lời khi giá X giảm
                        else: # Short Spread = Long X
                            pnl_x = (px - self.entry_price_x) * self.qty_x # Lời khi giá X tăng
                            
                        # Tổng bù trừ
                        net_pnl_usdt = pnl_y + pnl_x

                    spread_pct = abs(live_spread) / py
                    is_profitable = spread_pct >= SYSTEM_CONFIG['min_profit_pct']

                    # 3. Logic Tín Hiệu & QUẢN TRỊ RỦI RO
                    signal = self.current_position_state
                    exit_reason = "" 
                    
                    # --- KIỂM TRA P-VALUE ĐỂ FORCE EXIT ---
                    is_bad_cointegration = self.latest_p_value > SYSTEM_CONFIG['p_value_threshold']

                    # --- KIỂM TRA TIME STOP ---
                    is_time_out = False
                    if self.current_position_state != 'NEUTRAL' and self.entry_time:
                        elapsed_seconds = (datetime.now() - self.entry_time).total_seconds()
                        # Half life đơn vị là nến 15m -> đổi ra giây
                        max_seconds = self.latest_half_life * 15 * 60 * SYSTEM_CONFIG['time_stop_factor']                       
                        if elapsed_seconds > max_seconds:
                            is_time_out = True
                            time_msg = f"Time Limit ({elapsed_seconds/60:.0f}m > {max_seconds/60:.0f}m)"

                    # --- KIỂM TRA Z-SCORE STOPLOSS ---
                    is_statistical_stop = False
                    current_z_val = z_score # Lưu giá trị z hiện tại
                    
                    if self.current_position_state == 'LONG':
                        # Đang Long (kỳ vọng Z tăng lên), nhưng Z lại giảm sâu quá ngưỡng Stoploss âm
                        # Ví dụ: Entry lúc Z=-2.0, Stoploss thiết lập là 4.5 thì ngưỡng cắt là -4.5
                        if current_z_val < -SYSTEM_CONFIG['stop_loss_z']: 
                            is_statistical_stop = True                           
                    elif self.current_position_state == 'SHORT':
                        # Đang Short (kỳ vọng Z giảm xuống), nhưng Z lại tăng vọt quá ngưỡng Stoploss dương
                        if current_z_val > SYSTEM_CONFIG['stop_loss_z']:
                            is_statistical_stop = True

                    # --- KIỂM TRA HARD STOP ---
                    is_hard_stop = False
                    if self.current_position_state != 'NEUTRAL':
                        # Nếu lỗ vượt quá 50$ (net_pnl_usdt <= -50)
                        if net_pnl_usdt <= -SYSTEM_CONFIG['max_loss_usdt']:
                            is_hard_stop = True

                    # --- Logic Cắt Lệnh ---
                    if self.current_position_state != 'NEUTRAL':
                        # 1: HARD STOP (Cứu tài khoản trước tiên)
                        if is_hard_stop:
                            signal = 'NEUTRAL'
                            exit_reason = f"💸 MAX LOSS USDT: Lỗ quá {abs(net_pnl_usdt):.2f}$"
                        # 2. Ưu tiên cao nhất: Z-Score Stoploss (Cắt máu ngay lập tức)
                        elif is_statistical_stop:
                            signal = 'NEUTRAL'
                            exit_reason = f"💀 Z-SCORE STOPLOSS: Lệch chuẩn quá lớn (|Z| > {SYSTEM_CONFIG['stop_loss_z']})"                       
                        # 3. Ưu tiên nhì: P-Value (Mô hình hỏng)
                        elif is_bad_cointegration:
                            signal = 'NEUTRAL'
                            exit_reason = f"⚠️ FORCE EXIT: P-Value xấu ({self.latest_p_value:.4f})"                           
                        # 4. Ưu tiên ba: Time Stop (Hết giờ)
                        elif is_time_out:
                            signal = 'NEUTRAL'
                            exit_reason = f"⏳ TIME STOP: {time_msg}"
                    
                    # --- Logic Trading Bình Thường ---
                    if signal == self.current_position_state and not is_bad_cointegration: 
                        if self.current_position_state == 'NEUTRAL':
                            if z_score < -self.dynamic_entry_z and is_profitable: signal = 'LONG'
                            elif z_score > self.dynamic_entry_z and is_profitable: signal = 'SHORT'
                        
                        elif self.current_position_state == 'LONG':
                            if z_score >= SYSTEM_CONFIG['exit_z']: 
                                signal = 'NEUTRAL'
                                exit_reason = "Take Profit (Z-Score Reversion)"
                                
                        elif self.current_position_state == 'SHORT':
                            if z_score <= -SYSTEM_CONFIG['exit_z']: 
                                signal = 'NEUTRAL'
                                exit_reason = "Take Profit (Z-Score Reversion)"

                    # 4. Xử lý Hành động & Gửi Mail
                    if signal != self.current_position_state:
                        timestamp_str = datetime.now().strftime('%H:%M:%S %d/%m')
                        
                        # Nội dung HTML (Đã thêm P-Value)
                        html_body = f"""
                        <h3>BOT ALERT: {self.pair_name}</h3>
                        <p><b>Time:</b> {timestamp_str}</p>
                        <p><b>Action:</b> <span style="color:{'green' if signal=='LONG' else 'red'}; font-size:16px;"><b>{signal}</b></span></p>
                        <p><b>Z-Score:</b> {z_score:.4f}</p>
                        <p><b>Beta:</b> {calc_beta:.4f}</p>
                        <p><b>Current P-Value:</b> {self.latest_p_value:.4f}</p>
                        <p><b>Spread PnL:</b> {spread_pct*100:.2f}%</p>
                        """
                        if signal != 'NEUTRAL':
                            html_body += f"<p><b>Z-Threshold:</b> {self.dynamic_entry_z:.4f}</p>"

                        if exit_reason:
                            html_body += f"""
                            <p style='color:orange;'><b>Reason:</b> {exit_reason}</p>
                            <p><b>PnL:</b> {net_pnl_usdt:.2f}USDT</p>
                            """ 
                        html_body += "<hr><p><i>Auto Trading Bot</i></p>"

                        self.current_position_state = signal

                        if signal == 'LONG':
                            self.entry_time = datetime.now()
                            self.entry_price_y = py
                            self.entry_price_x = px
                            raw_qty_y = SYSTEM_CONFIG['fixed_loss_usdt'] / (spread_pct * py)
                            raw_qty_x = raw_qty_y * calc_beta

                            self.qty_y = self.normalize_amount(self.symbol_y, raw_qty_y)
                            self.qty_x = self.normalize_amount(self.symbol_x, raw_qty_x)
                            self.execute_bingx_order(self.symbol_y, 'buy', self.qty_y)
                            self.execute_bingx_order(self.symbol_x, 'sell', self.qty_x)

                            self.log(f"⚡ ENTRY LONG | Z: {z_score:.2f} | PnL%: {spread_pct*100:.2f}%", Fore.GREEN)
                            self.send_email(f"🟢 ENTRY LONG {self.pair_name}", html_body)
                            
                        elif signal == 'SHORT':
                            self.entry_time = datetime.now()
                            self.entry_price_y = py
                            self.entry_price_x = px
                            raw_qty_y = SYSTEM_CONFIG['fixed_loss_usdt'] / (spread_pct * py)
                            raw_qty_x = raw_qty_y * calc_beta

                            self.qty_y = self.normalize_amount(self.symbol_y, raw_qty_y)
                            self.qty_x = self.normalize_amount(self.symbol_x, raw_qty_x)
                            self.execute_bingx_order(self.symbol_y, 'sell', self.qty_y)
                            self.execute_bingx_order(self.symbol_x, 'buy', self.qty_x)

                            self.log(f"⚡ ENTRY SHORT | Z: {z_score:.2f} | PnL%: {spread_pct*100:.2f}%", Fore.RED)
                            self.send_email(f"🔴 ENTRY SHORT {self.pair_name}", html_body)

                        elif signal == 'NEUTRAL':
                            old_state = self.current_position_state

                            side_y = 'sell' if old_state == 'LONG' else 'buy'
                            side_x = 'buy' if old_state == 'LONG' else 'sell'
                            if self.qty_y > 0: self.execute_bingx_close(self.symbol_y, side_y, self.qty_y)
                            if self.qty_x > 0: self.execute_bingx_close(self.symbol_x, side_x, self.qty_x)
                            
                            log_color = Fore.RED if "FORCE EXIT" in exit_reason else Fore.YELLOW
                            
                            self.log(f"🏁 EXIT ({old_state}) | {exit_reason} | Z: {z_score:.2f}", log_color)
                            self.send_email(f"🟡 EXIT {self.pair_name}", html_body)
                            self.entry_time = None
                            self.entry_price_y = 0; self.entry_price_x = 0
                            self.qty_y = 0; self.qty_x = 0
                            self.re_calibrate()
                        
                    else:
                        # --- HEARTBEAT LOG ---
                        if SYSTEM_CONFIG['show_heartbeat']:
                            status_color = Fore.WHITE
                            if self.current_position_state == 'LONG': status_color = Fore.GREEN
                            elif self.current_position_state == 'SHORT': status_color = Fore.RED
                            
                            p_val_display = f"{self.latest_p_value:.4f}"
                            if self.latest_p_value > SYSTEM_CONFIG['p_value_threshold']:
                                p_val_display = f"{Fore.RED}{p_val_display}{Style.RESET_ALL}"
                            
                            print(f"{Fore.CYAN}[BOT {self.pair_name}]{Style.RESET_ALL} St: {status_color}{self.current_position_state:<5}{Style.RESET_ALL} | Z:{z_score:+.2f} | P-Val:{p_val_display}")

                    # [NEW] CƠ CHẾ TỰ HỦY (SELF-DESTRUCT)
                    # Nếu đang NEUTRAL (không giữ lệnh) VÀ P-Value xấu -> Dừng Bot
                    if self.current_position_state == 'NEUTRAL' and is_bad_cointegration:
                        self.log(f"🛑 STOPPING BOT: P-Value quá cao ({self.latest_p_value:.4f}). Hủy bot để giải phóng tài nguyên.", Fore.RED)
                        self.running = False # Break vòng lặp
                        break

                time.sleep(SYSTEM_CONFIG['bot_scan_interval'])
                
            except Exception as e:
                self.log(f"Error: {e}", Fore.RED)
                time.sleep(10)

    def stop(self):
        self.running = False

# ================= MODULE 3: INTELLIGENT SCANNER =================
class IntelligentScanner:
    def __init__(self):
        self.exchange = getattr(ccxt, SYSTEM_CONFIG['exchange'])({'enableRateLimit': True})
        self.active_bots = {} 

    def fetch_data(self, symbol):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=SYSTEM_CONFIG['timeframe'], limit=SYSTEM_CONFIG['limit_history'])
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df['close']
        except:
            return None

    def calculate_half_life(self, spread):
        try:
            spread_lag = spread.shift(1)
            spread_lag.iloc[0] = spread_lag.iloc[1]
            spread_ret = spread - spread_lag
            spread_ret.iloc[0] = spread_ret.iloc[1]
            spread_lag2 = sm.add_constant(spread_lag)
            model = sm.OLS(spread_ret, spread_lag2)
            res = model.fit()
            theta = res.params.iloc[1]
            if theta == 0: return 9999
            half_life = -np.log(2) / theta
            return max(1, half_life)
        except:
            return 9999

    def analyze_pair(self, asset_a, asset_b):
        s1 = self.fetch_data(asset_a)
        time.sleep(0.1) 
        s2 = self.fetch_data(asset_b)
        
        if s1 is None or s2 is None: return None
        
        df = pd.concat([s1, s2], axis=1, join='inner')
        df.columns = ['close_y', 'close_x']
        
        if len(df) < 500: return None

        try:
            _, p_value, _ = coint(np.log(df['close_y']), np.log(df['close_x']))
        except: return None

        x = sm.add_constant(np.log(df['close_x']))
        result = sm.OLS(np.log(df['close_y']), x).fit()
        gamma = result.params.iloc[1]
        
        spread = np.log(df['close_y']) - gamma * np.log(df['close_x'])
        half_life = self.calculate_half_life(spread)

        return {
            'pair_key': f"{asset_a}-{asset_b}",
            'symbol_y': asset_a,
            'symbol_x': asset_b,
            'p_value': p_value,
            'half_life': half_life,
            'data': df 
        }

    def run(self):
        print(f"{Fore.MAGENTA}=== HỆ THỐNG AUTO-TRADING ĐA CẶP (RISK CONTROL ENABLED) ===")
        print(f"Bắt đầu quét... (Interval: {SYSTEM_CONFIG['scan_interval']}s)")
        
        while True:
            print(f"\n{Fore.YELLOW}{'='*20} [SCANNER: {datetime.now().strftime('%H:%M:%S')}] {'='*20}{Style.RESET_ALL}")

            # 1. Dọn dẹp các Bot đã chết (Do tự hủy bên trên)
            dead_bots = [k for k, v in self.active_bots.items() if not v.is_alive()]
            for k in dead_bots:
                print(f"{Fore.RED}>>> REMOVING DEAD BOT: {k}{Style.RESET_ALL}")
                del self.active_bots[k]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(self.analyze_pair, p[0], p[1]) for p in SYSTEM_CONFIG['pairs_pool']]
                
                results_table = []
                
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if not res: continue
                    
                    pair_key = res['pair_key']
                    p_val = res['p_value']
                    hl = res['half_life']
                    
                    is_coint = p_val < SYSTEM_CONFIG['p_value_threshold']
                    status_str = "FAIL"
                    
                    # --- CẬP NHẬT TRẠNG THÁI CHO BOT ---
                    if pair_key in self.active_bots:
                        self.active_bots[pair_key].update_p_value_and_halflife(p_val, hl)
                        if not is_coint: 
                            status_str = f"{Fore.RED}CLOSING{Style.RESET_ALL}"
                        else:
                            status_str = f"{Fore.GREEN}RUNNING{Style.RESET_ALL}"
                    
                    # Logic khởi tạo Bot mới
                    if is_coint:
                        if pair_key not in self.active_bots:
                            status_str = "PASS -> START"
                            dynamic_z_window = int(hl * 1.5)
                            dynamic_z_window = max(20, min(dynamic_z_window, 200))
                            
                            print(f"{Fore.GREEN}>>> START BOT: {pair_key} (Z-Win: {dynamic_z_window}){Style.RESET_ALL}")
                            
                            new_bot = TradingBotWorker(
                                symbol_y=res['symbol_y'],
                                symbol_x=res['symbol_x'],
                                z_window=dynamic_z_window,
                                initial_history=res['data'],
                                initial_hl=hl
                            )
                            # Cập nhật ngay p_value đầu tiên
                            new_bot.update_p_value_and_halflife(p_val, hl)
                            new_bot.daemon = True 
                            new_bot.start()
                            self.active_bots[pair_key] = new_bot

                    results_table.append([pair_key, f"{p_val:.4f}", f"{hl:.1f}", status_str])

                print(tabulate(results_table, headers=["Pair", "P-Val", "H-Life", "Status"], tablefmt="simple"))
            
            print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
            
            # Kiểm tra Bot chết
            dead_bots = [k for k, v in self.active_bots.items() if not v.is_alive()]
            for k in dead_bots:
                del self.active_bots[k]

            time.sleep(SYSTEM_CONFIG['scan_interval'])

if __name__ == "__main__":
    try:
        system = IntelligentScanner()
        system.run()
    except KeyboardInterrupt:
        print("\nĐã dừng hệ thống.")