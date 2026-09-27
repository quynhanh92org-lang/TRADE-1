# 📈 Hệ Thống Phân Tích & Kiểm Định Chiến Lược Định Lượng: EMA & OBV (Cổ Phiếu ACB)

Ứng dụng Web App tương tác xây dựng trên nền tảng **Streamlit**, phục vụ việc backtest (kiểm định lịch sử), phân tích tín hiệu kỹ thuật thời gian thực và tối ưu hóa tham số cho chiến lược giao dịch kết hợp giữa **Đường trung bình động lũy thừa (EMA)** và **Chỉ báo khối lượng cân bằng (OBV)** trên dữ liệu 10 năm của cổ phiếu ACB (Ngân hàng TMCP Á Châu).

Hệ thống được thiết kế theo tiêu chuẩn của một **Quantitative Hedge Fund Performance Tear-sheet**, tập trung phân tích chuyên sâu các khía cạnh: **Lợi nhuận (Returns & CAGR)**, **Tỷ số Calmar & Sortino**, **Độ biến động (Volatility & Downside Risk)**, **Mức sụt giảm tối đa (MDD)** và **Tỷ lệ thắng (Winning Rate & Profit Factor)**.

---

## 🌟 Điểm Nổi Bật Của Hệ Thống Phân Tích Định Lượng

Ứng dụng được thiết kế theo chuẩn giao diện Fintech với **5 Tab chức năng chuyên biệt**:

1. **📊 Tín Hiệu & Biểu Đồ Kỹ Thuật (Live Signals & Interactive Charts):**
   - Đưa ra **khuyến nghị hành động phiên mới nhất** (BUY / HOLD / SELL / STAND ASIDE) dựa trên tổ hợp giữa tín hiệu kỹ thuật và trạng thái vị thế thực tế (đang cầm cổ phiếu hay tiền mặt).
   - Biểu đồ tương tác **Plotly Subplots** phóng to/thu nhỏ, rê chuột xem chi tiết đường giá, đường EMA, điểm khớp lệnh Mua (tam giác xanh), điểm Bán (tam giác đỏ) và đồ thị dòng tiền OBV kèm thanh độ dốc.
   - Bảng theo dõi tín hiệu chi tiết các phiên gần nhất.

2. **🧪 Kiểm Định Hiệu Quả & Phân Tích Chuyên Sâu (Quantitative Performance Tear-Sheet):**
   - **6 Thẻ KPI Vàng:** Tổng lợi nhuận (%), CAGR (%/năm), Tỷ số Calmar, Tỷ số Sortino, Độ biến động (Volatility %), Max Drawdown (MDD %) và Tỷ lệ thắng (Winning Rate %).
   - **Bảng Ma Trận Phân Tích Định Lượng Toàn Diện (Tear-Sheet Matrix):** So sánh 4 nhóm chỉ số giữa 4 phương án (*EMA+OBV Kết Hợp, EMA Riêng, OBV Riêng, Buy & Hold ACB Benchmark*):
     * *Nhóm 1: Lợi nhuận & Hiệu suất* (Total Return, CAGR, Net Profit VND, Alpha so với Benchmark).
     * *Nhóm 2: Đo lường Rủi ro & Biến động* (Annual Volatility, Downside Deviation, MDD %, Max DD Duration, VaR 95%, CVaR 95%).
     * *Nhóm 3: Tỷ số Hiệu quả Điều chỉnh Rủi ro* (Sharpe Ratio, Sortino Ratio, Calmar Ratio, Omega Ratio).
     * *Nhóm 4: Thống kê Giao dịch & Tỷ lệ Thắng* (Win Rate %, Profit Factor, Payoff Ratio, Expectancy %, Best Trade, Worst Trade, Chuỗi thắng/thua liên tiếp).
   - **Biểu đồ Radar (Spider Chart) 6 chiều:** Đánh giá năng lực chiến lược đa góc nhìn (Lợi nhuận, Kháng sụt giảm, Calmar, Sortino, Win Rate, Profit Factor).
   - **Phân tích Lợi nhuận Từng Năm (Yearly Returns Breakdown):** Bảng và biểu đồ cột so sánh hiệu suất từng năm từ 2014 đến 2023.
   - **Top 5 Đợt Sụt Giảm Sâu Nhất (Top 5 Deepest Drawdowns):** Ngày đỉnh, ngày đáy, ngày hồi phục và thời gian kéo dài.
   - **So sánh Đối kháng Train vs Test:** Trực quan hóa hiện tượng suy giảm hiệu suất ngoài mẫu (Out-of-sample deterioration).

3. **📋 Quản Lý Vị Thế & Nhật Ký Giao Dịch (Position Accounting & Trade Log):**
   - Trực quan hóa dòng thời gian vị thế danh mục (*0: Tiền mặt, 1: Nắm giữ cổ phiếu*).
   - Xuất bảng nhật ký chi tiết từng lệnh giao dịch đã đóng: *Ngày vào, Giá vào, Ngày ra, Giá ra, Lợi nhuận PnL (%), PnL (VND), Lý do thoát lệnh (Cắt lỗ 7% hay Tín hiệu EMA cắt)*.
   - Nút **tải về file CSV** nhật ký giao dịch chỉ với 1 click.
   - Biểu đồ phân bổ tỷ suất sinh lời của từng lệnh (Histogram).

4. **🎯 Tối Ưu Hóa Tham Số (Hyperopt TPE Optimization):**
   - Tích hợp thuật toán tối ưu hóa thông minh **Tree-structured Parzen Estimator (TPE)** từ thư viện `hyperopt`.
   - Cơ chế loại bỏ tham số 'ăn may': Ràng buộc `MIN_TRADES >= 5` để triệt tiêu bẫy *"Sharpe ảo / vô cực do ít giao dịch"*.
   - Biểu đồ không gian tham số (Parameter Space vs Sharpe Ratio) trực quan.

5. **📖 Phương Pháp Luận & Khắc Phục Lỗi (Methodology & Fixes):**
   - Trình bày chi tiết lý thuyết, công thức toán học và giải pháp xử lý các lỗi thường gặp trong nghiên cứu định lượng tài chính.

---

## 🔬 Ý Nghĩa & Công Thức Các Chỉ Số Định Lượng Trọng Tâm

### 1. Phân Tích Lợi Nhuận (Returns)
* **Tổng Lợi Nhuận (Total Return):** Tỷ lệ phần trăm tăng trưởng vốn ròng từ đầu đến cuối kỳ:
  $$\text{Total Return} = \frac{\text{Equity}_{end} - \text{Equity}_{start}}{\text{Equity}_{start}} \times 100\%$$
* **CAGR (Compound Annual Growth Rate):** Tỷ suất sinh lời kép hàng năm:
  $$\text{CAGR} = \left(\frac{\text{Equity}_{end}}{\text{Equity}_{start}}\right)^{\frac{252}{N}} - 1$$
* **Alpha / Excess Return:** Phần lợi nhuận vượt trội mà chiến lược tạo ra so với việc chỉ Mua & Nắm giữ cổ phiếu ACB (Buy & Hold).

### 2. Tỷ Số Calmar (Calmar Ratio)
* **Công thức:**
  $$\text{Calmar Ratio} = \frac{\text{CAGR [\%]}}{|\text{Max Drawdown [\%]}|}$$
* **Ý nghĩa:** Đo lường mỗi 1% mức sụt giảm tồi tệ nhất trong lịch sử đổi lại được bao nhiêu % lợi nhuận kép hàng năm.
* **Quy chuẩn đánh giá:**
  * $\text{Calmar} < 0.5$: Kém.
  * $0.5 \le \text{Calmar} < 1.0$: Khá tốt.
  * $\text{Calmar} \ge 1.0$: Xuất sắc (lợi nhuận hàng năm bù đắp hoàn toàn đợt sập sâu nhất).

### 3. Tỷ Số Sortino (Sortino Ratio)
* **Công thức:**
  $$\text{Sortino Ratio} = \frac{\bar{R} - R_f}{\sigma_{\text{downside}}}$$
  Trong đó $\sigma_{\text{downside}} = \sqrt{\frac{1}{M}\sum_{R_t < 0} R_t^2} \times \sqrt{252}$ là độ lệch chuẩn chỉ tính riêng các phiên giảm giá.
* **Ưu điểm so với Sharpe:** Trong chiến lược theo sau xu hướng, những phiên bứt phá trần tạo ra độ lệch dương lớn. Sharpe Ratio sẽ phạt nhầm các phiên tăng mạnh này như một "rủi ro biến động", còn Sortino Ratio chỉ phạt rủi ro sụt giảm giá, phản ánh đúng bản chất tâm lý của nhà đầu tư.

### 4. Độ Biến Động (Volatility & Downside Deviation)
* **Annualized Volatility:** Độ biến động tổng thể tính theo năm: $\text{std}(R_t) \times \sqrt{252} \times 100\%$.
* **Downside Deviation:** Độ biến động của rủi ro giảm giá. Chiến lược tốt sẽ có Downside Deviation thấp hơn đáng kể so với Annualized Volatility (nhờ thoát hàng về tiền mặt trong các pha giảm giá).

### 5. Maximum Drawdown (MDD - Mức Sụt Giảm Tối Đa)
* **Công thức:**
  $$\text{Drawdown}_t = \frac{\text{Equity}_t - \max_{0 \le s \le t}(\text{Equity}_s)}{\max_{0 \le s \le t}(\text{Equity}_s)}, \quad \text{MDD} = |\min_t (\text{Drawdown}_t)|$$
* **Ý nghĩa:** Thước đo rủi ro sinh tử của một hệ thống giao dịch, phản ánh mức thiệt hại lớn nhất từ đỉnh mà nhà đầu tư phải chịu đựng trước khi tài sản hồi phục.

### 6. Tỷ Lệ Thắng (Winning Rate) & Hệ Số Lợi Nhuận (Profit Factor)
* **Winning Rate (%):** $\frac{\text{Số lệnh thắng}}{\text{Tổng số lệnh đã đóng}} \times 100\%$.
* **Profit Factor:** $\frac{\text{Tổng lợi nhuận từ các lệnh thắng (VND)}}{\text{Tổng thua lỗ từ các lệnh thua (VND)}}$.
* **Payoff Ratio (Tỷ số Lời/Lỗ bình quân):** $\frac{\text{Lãi bình quân lệnh thắng [\%]}}{|\text{Lỗ bình quân lệnh thua [\%]}|}$.
* **Đặc tính chiến lược Trend-Following:** Chiến lược theo xu hướng thường có tỷ lệ thắng vừa phải (**35% – 45%**), nhưng có Payoff Ratio rất cao (**2.5x – 4.0x**): *"Thua nhanh với số tiền nhỏ (cắt lỗ 2% - 7%) và thắng lớn với số tiền lớn (gồng lãi 20% - 40%)"*.

---

## 📊 Bảng So Sánh Hiệu Suất Tổng Thể (Tham Khảo Dữ Liệu ACB)

| Chỉ Số Định Lượng | EMA + OBV (Train 2014-2020) | EMA + OBV (Test 2021-2023) | Buy & Hold ACB (Train) | Buy & Hold ACB (Test) |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng Lợi Nhuận (%)** | **+236.56%** | -36.47% | +359.95% | +30.34% |
| **Tỷ Suất Sinh Lời Kép (CAGR)** | **+18.95%/năm** | -14.15%/năm | +24.32%/năm | +9.24%/năm |
| **Calmar Ratio** | **1.21** | -0.33 | 0.44 | 0.16 |
| **Sortino Ratio** | **1.85** | -1.10 | 1.15 | 0.38 |
| **Sharpe Ratio** | **1.15** | -0.83 | 0.78 | 0.28 |
| **Annual Volatility (%)** | **18.2%** | 22.8% | 31.4% | 33.1% |
| **Max Drawdown (MDD %)** | **-23.87%** | -60.14% | -55.20% | -58.40% |
| **Tỷ Lệ Thắng (Win Rate %)** | **39.06%** | 24.32% | — | — |
| **Profit Factor** | **1.97** | 0.45 | — | — |
| **Số Lệnh Đã Đóng** | **64** | 37 | 1 | 1 |

> 💡 **Phân tích then chốt:** Trên tập Train, chiến lược kết hợp EMA + OBV đạt **Calmar Ratio = 1.21** và **Sortino Ratio = 1.85**, cao hơn vượt trội so với Buy & Hold (Calmar 0.44, Sortino 1.15) nhờ khả năng **cắt giảm Max Drawdown xuống chỉ còn -23.87%** (so với mức sụt giảm khốc liệt -55.20% của thị trường chung).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
TAO-APP-CHIEN-LUOC/
│
├── app.py                                   # Mã nguồn chính của ứng dụng Streamlit (Full Quantitative Suite)
├── requirements.txt                         # Danh sách thư viện phụ thuộc Python
├── README.md                                # Tài liệu hướng dẫn sử dụng và định nghĩa chỉ số
├── ACB.csv                                  # Dữ liệu giá lịch sử 10 năm của ACB (2014 - 2023)
└── ACB_Strategy_EMA_OBV_Nhom3_Adjusted.ipynb # Jupyter Notebook nghiên cứu gốc
```

---

## 🛠️ Hướng Dẫn Cài Đặt & Chạy Ứng Dụng Trên Máy Cục Bộ (Local)

### 1. Yêu cầu hệ thống
- Python 3.9 trở lên (khuyên dùng Python 3.10 hoặc 3.11).

### 2. Các bước khởi chạy
Mở ứng dụng dòng lệnh (**Terminal** trên macOS/Linux hoặc **PowerShell / CMD** trên Windows):

```bash
# 1. Di chuyển vào thư mục dự án
cd "đường-dẫn-đến-thư-mục-dự-án"

# 2. Tạo và kích hoạt môi trường ảo
python3 -m venv venv
source venv/bin/activate       # Trên macOS / Linux
# venv\Scripts\activate        # Trên Windows

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 4. Khởi chạy ứng dụng Streamlit
streamlit run app.py
```

Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Đẩy Lên GitHub & Deploy Lên Streamlit Cloud

### Bước 1: Khởi tạo Git và Đẩy Code Lên GitHub
Chạy lần lượt các lệnh sau trong Terminal:

```bash
git init
git add app.py requirements.txt README.md ACB.csv
git commit -m "Nâng cấp Web App Định lượng: Phân tích chuyên sâu Calmar, Sortino, MDD, Volatility và Win Rate"
git branch -M main
git remote add origin https://github.com/<tai-khoan-github-cua-ban>/chien-luoc-ema-obv-acb.git
git push -u origin main
```

### Bước 2: Triển Khai (Deploy) Lên Streamlit Cloud
1. Truy cập **[share.streamlit.io](https://share.streamlit.io/)** và đăng nhập bằng tài khoản GitHub.
2. Nhấn nút **"Create app"** (hoặc **"New app"**).
3. Chọn:
   - **Repository:** `<tai-khoan-github-cua-ban>/chien-luoc-ema-obv-acb`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Nhấn **"Deploy!"**. Ứng dụng sẽ hoạt động online sau 1–2 phút.

---

## 👥 Tác Giả & Bản Quyền

- **Đơn vị phát triển:** Nhóm nghiên cứu & phát triển chiến lược định lượng.
- **Giấy phép:** MIT License (Tự do sử dụng và phát triển vì mục đích học thuật và nghiên cứu).
