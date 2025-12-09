import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from colorama import Fore, Style, init
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl

# Khởi tạo màu console
init(autoreset=True)

# ================= CẤU HÌNH BOT (FINAL) =================
CONFIG = {
    'exchange': 'binance',
    'symbol_y': 'DOT/USDT', # Asset A (Coin biến động)
    'symbol_x': 'ATOM/USDT', # Asset B (Coin nền tảng)
    'timeframe': '15m',     # Khung tham chiếu thống kê
    
    # Kalman Filter Settings
    'delta': 1e-4,          
    'vt': 1e-3,             
    
    # Z-Score Settings
    'z_window': 30,         # Số lượng nến M15 dùng để tính Mean/Std
    'entry_z': 2.0,         # Ngưỡng vào lệnh
    'exit_z': 0.5,          # Ngưỡng thoát lệnh
    
    # === QUẢN LÝ RỦI RO & LỢI NHUẬN (MỚI) ===
    # Phí sàn + Trượt giá dự kiến ~ 0.3%. 
    # Ta cần biên độ lệnh tối thiểu phải > 0.4% mới bõ công vào lệnh.
    'min_profit_pct': 0.004, # 0.4% (0.004)
    # TỐI ƯU HÓA TỐC ĐỘ QUÉT
    'scan_interval': 60,    # 10 Giây quét 1 lần
    
    # === CẤU HÌNH GMAIL ===
    'email_enabled': True,
    'email_sender': 'vuongtinhkhac@gmail.com',      # <--- ĐIỀN EMAIL CỦA BẠN
    'email_password': 'eiow sbkd isqr rtnu',        # <--- ĐIỀN APP PASSWORD 16 KÝ TỰ
    'email_receiver': 'vuongtinhkhac@gmail.com',    # Gửi cho chính mình
}

# ================= CLASS KALMAN FILTER =================
class KalmanFilterReg:
    def __init__(self, delta=1e-4, vt=1e-3):
        self.delta = delta 
        self.vt = vt       
        self.x = np.zeros((2, 1)) # State [beta, alpha] tổng hợp dữ liệu đầu vào 
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

# ================= CLASS TRADING BOT =================
class PairTradingBot:
    def __init__(self):
        self.exchange = getattr(ccxt, CONFIG['exchange'])()
        self.kf = KalmanFilterReg(delta=CONFIG['delta'], vt=CONFIG['vt'])
        self.spread_history = [] 
        
        # Biến lưu trữ thống kê (Cache)
        self.cached_mean = 0
        self.cached_std = 0
        
        # Biến cờ đánh dấu nến đã xử lý
        self.last_processed_candle_ts = None
        
        # === QUẢN LÝ TRẠNG THÁI LỆNH (Tránh Spam Mail) ===
        # Các trạng thái: 'NEUTRAL', 'LONG', 'SHORT'
        self.current_position_state = 'NEUTRAL' 

    def send_email_alert(self, subject, body_html):
        """Hàm gửi Email tối ưu"""
        if not CONFIG['email_enabled']: return

        try:
            msg = MIMEMultipart()
            msg['From'] = CONFIG['email_sender']
            msg['To'] = CONFIG['email_receiver']
            msg['Subject'] = subject
            msg.attach(MIMEText(body_html, 'html'))

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(CONFIG['email_sender'], CONFIG['email_password'])
                server.send_message(msg)
            
            print(f"{Fore.YELLOW}📧 [EMAIL SENT] Đã gửi mail thông báo: {subject}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Lỗi gửi Email: {e}{Style.RESET_ALL}")

    def fetch_history(self, limit=1000):
        """Warm-up: Khởi tạo dữ liệu từ quá khứ"""
        print(f"{Fore.YELLOW}>>> Đang tải lịch sử {CONFIG['timeframe']} để huấn luyện Bot...")
        try:
            ohlcv_y = self.exchange.fetch_ohlcv(CONFIG['symbol_y'], CONFIG['timeframe'], limit=limit)
            df_y = pd.DataFrame(ohlcv_y, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            ohlcv_x = self.exchange.fetch_ohlcv(CONFIG['symbol_x'], CONFIG['timeframe'], limit=limit)
            df_x = pd.DataFrame(ohlcv_x, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            
            df = pd.merge(df_y[['ts', 'c']], df_x[['ts', 'c']], on='ts', suffixes=('_y', '_x'))
            
            for idx, row in df.iterrows():
                beta, alpha, spread = self.kf.update(row['c_y'], row['c_x'])
                self.spread_history.append(spread)
            
            if len(self.spread_history) > CONFIG['z_window']:
                self.spread_history = self.spread_history[-CONFIG['z_window']:]

            series = pd.Series(self.spread_history)
            self.cached_mean = series.mean()
            self.cached_std = series.std()

            last_ts = pd.to_datetime(df.iloc[-1]['ts'], unit='ms')
            self.last_processed_candle_ts = (last_ts.hour * 60 + last_ts.minute) // 15

            print(f"{Fore.GREEN}✔ Khởi tạo xong! Beta: {beta:.4f} | Mean: {self.cached_mean:.4f} | Std: {self.cached_std:.4f}")
            return True
        except Exception as e:
            print(f"{Fore.RED}❌ Lỗi tải dữ liệu lịch sử: {e}")
            return False

    def fetch_current_price(self):
        try:
            ticker_y = self.exchange.fetch_ticker(CONFIG['symbol_y'])
            ticker_x = self.exchange.fetch_ticker(CONFIG['symbol_x'])
            return ticker_y['last'], ticker_x['last']
        except Exception as e:
            print(f"{Fore.RED}Lỗi kết nối API: {e}")
            return None, None

    def check_and_send_signal(self,is_profitable, z_score, price_y, price_x, beta):
        """Logic kiểm tra tín hiệu và gửi mail khi thay đổi trạng thái"""
        timestamp = datetime.now().strftime('%H:%M:%S %d/%m')
        signal_type = None
        action_msg = ""
        color_log = Fore.WHITE

        # 1. Xác định tín hiệu hiện tại
        if z_score < -CONFIG['entry_z'] and is_profitable:
            signal_type = 'LONG' # Mua Y, Bán X
        elif z_score > CONFIG['entry_z'] and is_profitable:
            signal_type = 'SHORT' # Bán Y, Mua X
        elif abs(z_score) < CONFIG['exit_z']: # Thoát lệnh khi về gần 0
            signal_type = 'NEUTRAL'
        else:
            signal_type = self.current_position_state # Giữ nguyên trạng thái cũ (Vùng chờ)

        # 2. So sánh với trạng thái cũ để quyết định gửi mail
        # Chỉ gửi khi có sự thay đổi TRẠNG THÁI quan trọng
        if signal_type != self.current_position_state:
            
            # Case 1: Vào lệnh LONG mới
            if signal_type == 'LONG' and self.current_position_state == 'NEUTRAL':
                action_msg = f"🟢 ENTRY LONG SPREAD (Mua {CONFIG['symbol_y']} / Bán {CONFIG['symbol_x']})"
                self.trigger_email("LONG ENTRY", z_score, price_y, price_x, beta, action_msg)
                self.current_position_state = 'LONG'
                color_log = Fore.GREEN

            # Case 2: Vào lệnh SHORT mới
            elif signal_type == 'SHORT' and self.current_position_state == 'NEUTRAL':
                action_msg = f"🔴 ENTRY SHORT SPREAD (Bán {CONFIG['symbol_y']} / Mua {CONFIG['symbol_x']})"
                self.trigger_email("SHORT ENTRY", z_score, price_y, price_x, beta, action_msg)
                self.current_position_state = 'SHORT'
                color_log = Fore.RED

            # Case 3: Thoát lệnh (Take Profit) từ LONG
            elif signal_type == 'NEUTRAL' and self.current_position_state == 'LONG':
                action_msg = f"🟡 TAKE PROFIT / EXIT LONG (Z-score về 0)"
                self.trigger_email("EXIT SIGNAL", z_score, price_y, price_x, beta, action_msg)
                self.current_position_state = 'NEUTRAL'
                color_log = Fore.YELLOW

            # Case 4: Thoát lệnh (Take Profit) từ SHORT
            elif signal_type == 'NEUTRAL' and self.current_position_state == 'SHORT':
                action_msg = f"🟡 TAKE PROFIT / EXIT SHORT (Z-score về 0)"
                self.trigger_email("EXIT SIGNAL", z_score, price_y, price_x, beta, action_msg)
                self.current_position_state = 'NEUTRAL'
                color_log = Fore.YELLOW

        return signal_type, color_log

    def trigger_email(self, type_title, z_score, py, px, beta, note):
        """Tạo nội dung HTML đẹp mắt"""
        subject = f"🔔 BOT ALERT: {type_title} | Z: {z_score:.2f}"
        
        color = "black"
        if "LONG" in type_title: color = "green"
        elif "SHORT" in type_title: color = "red"
        elif "EXIT" in type_title: color = "#D4AC0D" # Vàng đậm

        html = f"""
        <html>
            <body>
                <h2 style="color:{color};">{type_title} DETECTED</h2>
                <p><b>Thời gian:</b> {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}</p>
                <table style="border-collapse: collapse; width: 100%; max-width: 500px;">
                    <tr style="background-color: #f2f2f2;"><td style="padding:8px; border:1px solid #ddd;"><b>Cặp Giao Dịch</b></td><td style="padding:8px; border:1px solid #ddd;">{CONFIG['symbol_y']} - {CONFIG['symbol_x']}</td></tr>
                    <tr><td style="padding:8px; border:1px solid #ddd;"><b>Z-Score</b></td><td style="padding:8px; border:1px solid #ddd;"><b>{z_score:.4f}</b></td></tr>
                    <tr><td style="padding:8px; border:1px solid #ddd;"><b>Hedge Ratio (Beta)</b></td><td style="padding:8px; border:1px solid #ddd;">{beta:.4f}</td></tr>
                    <tr><td style="padding:8px; border:1px solid #ddd;"><b>Giá {CONFIG['symbol_y']}</b></td><td style="padding:8px; border:1px solid #ddd;">{py}</td></tr>
                    <tr><td style="padding:8px; border:1px solid #ddd;"><b>Giá {CONFIG['symbol_x']}</b></td><td style="padding:8px; border:1px solid #ddd;">{px}</td></tr>
                    <tr><td style="padding:8px; border:1px solid #ddd;"><b>Hành động</b></td><td style="padding:8px; border:1px solid #ddd; color:{color};"><b>{note}</b></td></tr>
                </table>
                <p><i>Bot Trading chạy trên VPS Windows.</i></p>
            </body>
        </html>
        """
        self.send_email_alert(subject, html)

    def run(self):
        if not self.fetch_history(): return

        print(f"\n{Fore.CYAN}=== BOT ĐANG CHẠY (QUÉT {CONFIG['scan_interval']}s/lần) ===")
        print(f"Chế độ: Hybrid (Thống kê M15 - Tín hiệu Realtime)")
        print(f"Email Alerts: {'BẬT' if CONFIG['email_enabled'] else 'TẮT'}")
        print("-" * 70)
        
        while True:
            try:
                # Logic này giúp Bot luôn chạy ngay khi nến M1 vừa đóng
                now = datetime.now()
                sleep_to_next_minute = 60 - now.second + 1 # +1 giây đệm để sàn kịp chốt nến
                time.sleep(sleep_to_next_minute)

                price_y, price_x = self.fetch_current_price()
                
                if price_y and price_x:
                    # 1. Logic Update Thống kê (Mỗi 15 phút)
                    #now = datetime.now()
                    current_candle_ts = (now.hour * 60 + now.minute) // 15
                    
                    if self.last_processed_candle_ts is not None and current_candle_ts != self.last_processed_candle_ts:
                        print(f"{Fore.MAGENTA}\n>>> [NEW CANDLE] Update thống kê M15...")
                        if self.current_position_state == 'NEUTRAL':
                            beta_new, _, spread_new = self.kf.update(price_y, price_x)
                            self.spread_history.append(spread_new)
                            if len(self.spread_history) > CONFIG['z_window']:
                                self.spread_history.pop(0)
                            series = pd.Series(self.spread_history)
                            self.cached_mean = series.mean()
                            self.cached_std = series.std()
                            print(f"    Updated Beta: {beta_new:.4f} ")
                        else:
                            print(f"    [FREEZE] Đang gồng lệnh {self.current_position_state} -> Giữ nguyên Beta & Mean/Std cũ để tham chiếu.")                      
                        
                        self.last_processed_candle_ts = current_candle_ts
                    # 2. Logic Tính toán Tín hiệu (Realtime)
                    current_beta = self.kf.x[0, 0]
                    current_alpha = self.kf.x[1, 0]
                    live_spread = price_y - (current_beta * price_x + current_alpha)
                    spread_pct = abs(live_spread) / price_y
                    is_profitable = spread_pct >= CONFIG['min_profit_pct']
                    
                    if self.cached_std == 0: z_score = 0
                    else: z_score = (live_spread - self.cached_mean) / self.cached_std
                    
                    # Lưu trạng thái cũ trước khi kiểm tra tín hiệu
                    previous_state = self.current_position_state
                    # 3. Kiểm tra Tín hiệu và Gửi Mail (Tránh Spam)
                    signal_now, color_log = self.check_and_send_signal(is_profitable, z_score, price_y, price_x, current_beta)
                    # === LOGIC MỚI: RE-TRAIN SAU KHI THOÁT LỆNH ===
                    # Nếu trạng thái chuyển từ CÓ LỆNH (Long/Short) -> VỀ KHÔNG (Neutral)
                    if previous_state != 'NEUTRAL' and signal_now == 'NEUTRAL':
                        print(f"\n{Fore.CYAN}>>> [RESET] Đã thoát lệnh. Tiến hành Re-train lại Bot với dữ liệu mới nhất...{Style.RESET_ALL}")
                        
                        # Gọi lại hàm fetch_history để làm mới hoàn toàn Beta, Mean, Std, Spread History
                        # Dựa trên 1000 nến gần nhất (bao gồm cả những nến vừa bị bỏ qua lúc gồng lệnh)
                        is_ready = self.fetch_history()
                        
                        if is_ready:
                            print(f"{Fore.GREEN}>>> Re-train hoàn tất! Bot đã sẵn sàng cho cơ hội mới.{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}>>> Re-train thất bại! Bot sẽ thử lại ở vòng lặp sau.{Style.RESET_ALL}")
                    # In Log gọn gàng
                    timestamp_str = datetime.now().strftime('%H:%M:%S')
                    status_display = f"{color_log}{signal_now} (State: {self.current_position_state}){Style.RESET_ALL}"
                    print(f"\r[{timestamp_str}] Beta:{current_beta:.3f} | Z:{z_score:.3f} | {status_display}", end="")
                    
                #time.sleep(CONFIG['scan_interval'])
                
            except KeyboardInterrupt:
                print("\n\nĐã dừng Bot.")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Lỗi Runtime: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = PairTradingBot()
    bot.run()