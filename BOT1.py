import ccxt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import time
from datetime import datetime
from tabulate import tabulate
from colorama import Fore, Style, init
import concurrent.futures
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl

# Khởi tạo màu cho Window Console
init(autoreset=True)

# ================= CẤU HÌNH BOT =================
CONFIG = {
    'exchange': 'binance',      # Sàn giao dịch
    'timeframe': '15m',         # Khung thời gian
    'limit': 1000,              # Số lượng nến (Mẫu dữ liệu)
    'scan_interval': 300,       # Quét lại sau mỗi 300 giây (5 phút)
    'p_value_threshold': 0.1,  # Ngưỡng P-value tối đa
    'halflife_threshold': 50,   # Ngưỡng Half-life tối đa
    
    # === CẤU HÌNH GMAIL ===
    'email_enabled': True,                  # Bật/Tắt gửi mail
    'email_sender': 'vuongtinhkhac@gmail.com',  # <--- ĐIỀN EMAIL CỦA BẠN
    'email_password': 'eiow sbkd isqr rtnu',    # <--- ĐIỀN MẬT KHẨU ỨNG DỤNG (APP PASSWORD) 16 KÝ TỰ
    'email_receiver': 'vuongtinhkhac@gmail.com',# Gửi cho chính mình

    'email_cooldown': 3600, # (Giây) 3600s = 60 phút. Không gửi lại mail cho cùng 1 cặp trong thời gian này.
    # DANH SÁCH CÁC CẶP TÀI SẢN MUỐN QUÉT (CẶP CỦA CẶP)
    # Bạn có thể thêm bất kỳ cặp nào vào đây
    'pairs_pool': [
        ('BNB/USDT', 'ETH/USDT'),
        ('DOGE/USDT', 'SHIB/USDT'),
        ('LTC/USDT', 'BCH/USDT'),
        ('DOT/USDT', 'ATOM/USDT'),
        ('ETC/USDT', 'ETH/USDT'),
        ('ARB/USDT', 'OP/USDT'), # Layer 2 pairs
        ('SOL/USDT', 'JUP/USDT'),
    ]
}

class CointegrationScanner:
    def __init__(self):
        self.exchange = getattr(ccxt, CONFIG['exchange'])()
        self.exchange.enableRateLimit = True
        self.last_alert_times = {}

    def fetch_data(self, symbol):
        """Lấy dữ liệu nến từ sàn (Public API)"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=CONFIG['timeframe'], limit=CONFIG['limit'])
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df['close']
        except Exception as e:
            # print(f"Lỗi tải {symbol}: {e}") # Bỏ comment nếu muốn debug
            return None

    def calculate_half_life(self, spread):
        """Tính Half-Life của chuỗi Spread"""
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
            return max(0, half_life) # Tránh số âm
        except:
            return 9999

    def analyze_pair(self, asset_a, asset_b):
        """Phân tích 1 cặp: Tính P-value và Half-Life"""
        # Lấy dữ liệu
        series_a = self.fetch_data(asset_a)
        time.sleep(0.1) # Nghỉ nhẹ để tránh rate limit nếu chạy đơn luồng
        series_b = self.fetch_data(asset_b)

        if series_a is None or series_b is None:
            return None

        # Đồng bộ dữ liệu (Chỉ lấy phần giao nhau)
        df = pd.concat([series_a, series_b], axis=1, join='inner')
        if len(df) < CONFIG['limit'] * 0.9: # Yêu cầu đủ ít nhất 90% dữ liệu
            return None

        S1 = np.log(df.iloc[:, 0]) # Log price Asset A
        S2 = np.log(df.iloc[:, 1]) # Log price Asset B

        # 1. Engle-Granger Cointegration Test
        try:
            # statsmodels coint trả về: score, pvalue, critical_values
            _, p_value, _ = coint(S1, S2)
        except:
            return None

        # 2. Tính Spread để tính Half-Life
        # Hồi quy tuyến tính để tìm Hedge Ratio (gamma) sơ bộ
        x = sm.add_constant(S2)
        result = sm.OLS(S1, x).fit()
        gamma = result.params.iloc[1]
        spread = S1 - gamma * S2

        # 3. Tính Half-Life
        halflife = self.calculate_half_life(spread)

        return {
            'pair': f"{asset_a} - {asset_b}",
            'p_value': p_value,
            'half_life': halflife,
            'hedge_ratio': gamma
        }

    def send_notification_email(self, valid_results):
        if not CONFIG['email_enabled']: return

        try:
            sender_email = CONFIG['email_sender']
            receiver_email = CONFIG['email_receiver']
            password = CONFIG['email_password']

            # Tạo nội dung HTML cho Email
            html_table_rows = ""
            for res in valid_results:
                html_table_rows += f"""
                <tr>
                    <td style="padding:8px; border:1px solid #ddd;"><b>{res['pair']}</b></td>
                    <td style="padding:8px; border:1px solid #ddd; color:green;">{res['p_value']:.5f}</td>
                    <td style="padding:8px; border:1px solid #ddd;">{res['half_life']:.2f}</td>
                    <td style="padding:8px; border:1px solid #ddd;">{res['hedge_ratio']:.3f}</td>
                </tr>
                """

            html_content = f"""
            <html>
                <body>
                    <h2>🚀 Phát hiện {len(valid_results)} Cặp Giao Dịch Tiềm Năng</h2>
                    <p>Thời gian quét: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                    <table style="border-collapse: collapse; width: 100%;">
                        <tr style="background-color: #f2f2f2;">
                            <th style="padding:8px; border:1px solid #ddd; text-align:left;">Cặp (Pair)</th>
                            <th style="padding:8px; border:1px solid #ddd; text-align:left;">P-Value</th>
                            <th style="padding:8px; border:1px solid #ddd; text-align:left;">Half-Life</th>
                            <th style="padding:8px; border:1px solid #ddd; text-align:left;">Hedge Ratio</th>
                        </tr>
                        {html_table_rows}
                    </table>
                    <p><i>Bot chạy tự động từ VPS Windows.</i></p>
                </body>
            </html>
            """

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = receiver_email
            msg['Subject'] = f"🔔 BOT ALERT: Tìm thấy {len(valid_results)} cặp Coin - {datetime.now().strftime('%H:%M')}"
            msg.attach(MIMEText(html_content, 'html'))

            # Kết nối an toàn tới Gmail
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(sender_email, password)
                server.send_message(msg)
            
            print(f"{Fore.YELLOW}📧 Đã gửi Email thông báo thành công!{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}❌ Lỗi gửi Email: {e}{Style.RESET_ALL}")

    def run(self):
        print(f"{Fore.CYAN}=== SCANNER ===")
        print(f"Cooldown: {CONFIG['email_cooldown']}s (Không gửi lặp lại trong thời gian này)")
        
        while True:
            start_time = time.time()
            results = []
            current_timestamp = time.time()
            
            print(f"\n{Fore.YELLOW}>>> Đang quét dữ liệu... ({datetime.now().strftime('%H:%M:%S')})")

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_pair = {executor.submit(self.analyze_pair, p[0], p[1]): p for p in CONFIG['pairs_pool']}
                for future in concurrent.futures.as_completed(future_to_pair):
                    data = future.result()
                    if data: results.append(data)

            results.sort(key=lambda x: x['p_value'])

            table_data = []
            pairs_to_notify = [] # Danh sách CHUẨN BỊ gửi mail

            for res in results:
                p_val = res['p_value']
                hl = res['half_life']
                pair_name = res['pair']
                
                is_coint = p_val < CONFIG['p_value_threshold']
                is_fast = hl < CONFIG['halflife_threshold']
                
                status = "FAIL"
                color = Fore.WHITE
                
                if is_coint and is_fast:
                    status = "✅ GOOD"
                    color = Fore.GREEN
                    
                    # === LOGIC CHỐNG SPAM Ở ĐÂY ===
                    last_sent = self.last_alert_times.get(pair_name, 0)
                    
                    # Nếu chưa từng gửi HOẶC đã quá thời gian cooldown
                    if (current_timestamp - last_sent) > CONFIG['email_cooldown']:
                        pairs_to_notify.append(res)
                        # Cập nhật thời gian gửi mới nhất luôn (tạm tính là sẽ gửi thành công)
                        self.last_alert_times[pair_name] = current_timestamp
                    else:
                        # Vẫn in ra màn hình nhưng đánh dấu là đã gửi rồi
                        status = "✅ SENT (Cooling)"
                        
                elif is_coint:
                    status = "⚠️ SLOW"
                    color = Fore.CYAN
                else:
                    status = "❌ NO COINT"
                    color = Fore.RED

                table_data.append([
                    color + pair_name + Style.RESET_ALL,
                    f"{p_val:.5f}", f"{hl:.2f}", f"{res['hedge_ratio']:.3f}", status
                ])

            print(f"Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            print(tabulate(table_data, headers=["Pair", "P-Val", "H-Life", "Hedge", "Status"], tablefmt="grid"))

            # === GỬI EMAIL CHỈ CHO CÁC CẶP MỚI HOẶC HẾT COOLDOWN ===
            if len(pairs_to_notify) > 0:
                print(f"{Fore.GREEN}🎯 Phát hiện {len(pairs_to_notify)} thông báo mới cần gửi...{Style.RESET_ALL}")
                self.send_notification_email(pairs_to_notify)
            else:
                print("Không có thông báo mới (Các cặp tốt đang trong thời gian chờ Cooldown).")

            elapsed = time.time() - start_time
            sleep_time = max(0, CONFIG['scan_interval'] - elapsed)
            print(f"{Fore.MAGENTA}Chờ {int(sleep_time)}s...{Style.RESET_ALL}")
            time.sleep(sleep_time)

if __name__ == "__main__":
    try:
        scanner = CointegrationScanner()
        scanner.run()
    except KeyboardInterrupt:
        print("\nĐã dừng Bot.")
    except Exception as e:
        print(f"Lỗi Fatal: {e}")