import ccxt
import json
import time

# --- CẤU HÌNH (Nên để trong file .env hoặc config riêng) ---
# LƯU Ý: Thay thế bằng Key MỚI của bạn, key cũ đã bị lộ!
API_KEY = '6xfkNHeeto3IgZ4IcN9oIcMGxte0hmt2mWAb4tTnuuEqnkwGymQzV4KX6jEULn6sHiUeMGIK5iUUPZUUrcw'
SECRET_KEY = 'Qa9mcqlj3ashYlRHeGLCC2Isaq8fj6CSyBIG3GLnk9GWQis9OHdKuAbTl8ewSow0k1wNn3y3I30N7yIJaUg'
SYMBOL = 'BTC/USDT:USDT'  # Định nghĩa cặp giao dịch tại 1 chỗ

# Khởi tạo instance CCXT một lần duy nhất cho Swap (Perpetual)
exchange = ccxt.bingx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # Mặc định là Swap để trade bot
        'adjustForTimeDifference': True
    }
})

def check_balance():
    """Kiểm tra số dư USDT trong ví Swap (Perpetual)"""
    print("--- 🔍 ĐANG KIỂM TRA VÍ SWAP ---")
    try:
        balance = exchange.fetch_balance()
        usdt_free = balance['USDT']['free'] if 'USDT' in balance else 0
        
        print(f"✅ VÍ PERPETUAL (Dùng cho Bot): {usdt_free} USDT")
        
        if usdt_free < 2: # BingX thường yêu cầu lệnh tối thiểu > 2-5 USDT
            print("⚠️ CẢNH BÁO: Số dư quá thấp để đặt lệnh an toàn.")
        
        return usdt_free
    except Exception as e:
        print(f"❌ Lỗi kiểm tra ví: {e}")
        return 0

def set_leverage(symbol, leverage):
    """Cài đặt đòn bẩy"""
    try:
        # BingX yêu cầu set margin mode trước hoặc set leverage trực tiếp
        # Code này set leverage cho thị trường cụ thể
        exchange.set_leverage(leverage, symbol, {'side': 'LONG'})
        print(f"✅ Đã set đòn bẩy x{leverage} cho {symbol}")
    except Exception as e:
        print(f"❌ Lỗi set đòn bẩy: {e}")

def execute_bingx_order(symbol, side, amount_usdt):
    try:
        
        # Tính số lượng coin từ số USDT muốn đi lệnh
        amount_coin = amount_usdt / 90000
        
        # Chuẩn hóa số lượng theo quy định sàn (tránh lỗi precision)
        amount_final = exchange.amount_to_precision(symbol, amount_coin)
        params = {}
        if side == 'buy':
            params['positionSide'] = 'LONG'  # Mua là mở Long
        elif side == 'sell':
            params['positionSide'] = 'SHORT' # Bán là mở Short
        order = exchange.create_order(symbol, 'market', side, float(amount_final), params=params)
        
        print(f"✅ ĐẶT LỆNH MỞ THÀNH CÔNG: ID {order['id']}")
        return order
        
    except Exception as e:
        print(f"❌ LỖI ĐẶT LỆNH: {e}")

def execute_close_order(symbol, side, amount_usdt):
    try:
        
        # Tính số lượng coin từ số USDT muốn đi lệnh
        amount_coin = amount_usdt / 90000
        
        # Chuẩn hóa số lượng theo quy định sàn (tránh lỗi precision)
        amount_final = exchange.amount_to_precision(symbol, amount_coin)
        params = {}
        if side == 'buy':
            params['positionSide'] = 'SHORT'  # Mua là mở Long
        elif side == 'sell':
            params['positionSide'] = 'LONG' # Bán là mở Short
        
        order = exchange.create_order(symbol, 'market', side, float(amount_final), params=params)
        
        print(f"✅ ĐẶT LỆNH ĐÓNG THÀNH CÔNG: ID {order['id']}")
        return order
        
    except Exception as e:
        print(f"❌ LỖI ĐẶT LỆNH: {e}")

if __name__ == "__main__":
    try:
        # 1. Load thị trường để lấy thông tin precision
        print("⏳ Đang tải thông tin thị trường...")
        exchange.load_markets()
        
        # 2. Kiểm tra ví
        current_balance = check_balance()
        
        if current_balance > 0:
            # 3. Cài đòn bẩy
            set_leverage(SYMBOL, 40)
            
            # 4. Đặt lệnh (Ví dụ: Mua 10 USDT tiền BTC)
            # Lưu ý: BingX có yêu cầu min volume (thường khoảng 2-5 USDT)
            print("--- 🚀 ĐẶT LỆNH MẪU TRÊN BINGX ---")
            execute_bingx_order(SYMBOL, 'buy', 10) 
            time.sleep(5) 
            print("--- 🚀 ĐẶT LỆNH MẪU ĐÓNG TRÊN BINGX ---")
            execute_close_order(SYMBOL, 'sell', 10)
        else:
            print("⛔ Dừng bot: Không có số dư trong ví Perpetual.")
            
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")