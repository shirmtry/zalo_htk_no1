import os
import json
import sqlite3
import pandas as pd
import torch
import re
import unicodedata
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN CHUẨN CỦA BTC ZALO
# ==========================================
INPUT_DIR = "/data"       # Nơi BTC mount dữ liệu câu hỏi vào
OUTPUT_DIR = "/result"    # Nơi BTC yêu cầu lưu file submission.jsonl
MODEL_PATH = "/code/qwen_sql_merged_4bit" # Model nằm trong thư mục /code

# Đảm bảo thư mục output tồn tại trước khi ghi file
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. CÁC HÀM SQLITE TÙY CHỈNH
# ==========================================
def vn_norm(text):
    if text is None: return None
    text = str(text).lower().replace("đ", "d")
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def vn_ym(date_str):
    if date_str is None: return None
    match = re.search(r'(\d{4})[-/](\d{2})', str(date_str))
    return match.group(1) + match.group(2) if match else None

def vn_ymd(date_str):
    if date_str is None: return None
    match = re.search(r'(\d{4})[-/](\d{2})[-/](\d{2})', str(date_str))
    return match.group(1) + match.group(2) + match.group(3) if match else None

# ==========================================
# 3. LOAD MODEL OFFLINE
# ==========================================
print("⏳ Đang load Model Offline...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    device_map="auto", 
    dtype=torch.float16, # Đã sửa từ torch_dtype thành dtype để hết warning
    trust_remote_code=True
)
model.eval()

# ==========================================
# 4. SYSTEM PROMPT (Giữ nguyên bản dài của bạn)
# ==========================================
system_prompt = """Bạn là chuyên gia SQLite phân tích dữ liệu kinh doanh tiếng Việt.
Từ câu hỏi, README và schema được cung cấp, hãy viết ĐÚNG MỘT câu SELECT để trả lời.
CHỈ trả về một JSON hợp lệ, không markdown, không giải thích ngoài JSON:
{"sql": "SELECT ...", "type": "number|percent|list|table|date|string|boolean|unanswerable", "reason": "ngắn gọn"}
Nếu không thể trả lời từ dữ liệu: {"sql": null, "type": "unanswerable", "reason": "..."}

QUY TẮC BẮT BUỘC
1. Chỉ dùng SELECT/WITH. Không INSERT/UPDATE/DELETE/DDL/PRAGMA. Luôn bọc tên bảng và cột trong dấu ngoặc kép: "ten_bang", "ten_cot".
2. Chỉ dùng bảng/cột có thật trong schema của doanh nghiệp này. Không bịa tên.
3. DỮ LIỆU XUẤT EXCEL CÓ 2 ĐẶC ĐIỂM PHẢI XỬ LÝ ĐÚNG THỨ TỰ:
   (a) DÒNG TIẾP NỐI: dòng chỉ có tên hàng/số lượng/tiền, còn mã đơn + ngày + trạng thái + khách là NULL —
       dòng đó thuộc ĐƠN PHÍA TRÊN. BẮT BUỘC áp dụng forward-fill cho MỌI bảng chi tiết bán hàng, KHÔNG có
       ngoại lệ và KHÔNG được lược bỏ kể cả khi bạn không thấy dòng NULL. Nếu lược bỏ bước điền, việc lọc
       trạng thái/ngày trên cột gốc sẽ XÓA MẤT các dòng tiếp nối (NULL) → thiếu doanh thu.
       Tuyệt đối KHÔNG lọc "ma_don IS NOT NULL" và không GROUP BY/HAVING trên mã đơn gốc trước khi điền.
       Mẫu (đổi tên bảng/cột theo schema):
         WITH s AS (SELECT rowid rid, <cot_ma_don> oid, <cot_ngay> ngay, <cot_trang_thai> st
                    FROM "<ten_bang>" WHERE <cot_ma_don> IS NOT NULL),
         g AS (SELECT rowid rid, MAX(CASE WHEN <cot_ma_don> IS NOT NULL THEN rowid END)
                          OVER (ORDER BY rowid) grp FROM "<ten_bang>"),
         f AS (SELECT t.*, s.oid f_oid, s.ngay f_ngay, s.st f_st
               FROM g JOIN s ON s.rid = g.grp JOIN "<ten_bang>" t ON t.rowid = g.rid),
         d AS (SELECT DISTINCT * FROM f)
         SELECT SUM(<cot_tien>) FROM d
         WHERE <cot_so_luong> IS NOT NULL AND f_oid ... AND f_ngay ...
       (thay <...> bằng tên cột THẬT trong schema; ví dụ biz_105 là "ma_hoa_don","ngay_giao_dich","status","hang","so_luong","tri_gia")
       Lọc theo đơn/ngày/trạng thái dùng f_oid/f_ngay/f_st (đã điền), KHÔNG dùng cột gốc còn NULL.
   (b) KHỬ TRÙNG LẶP TOÀN PHẦN: bảng chi tiết xuất Excel rất hay có dòng lặp y nguyên trên MỌI cột (lỗi xuất file),
       kể cả khi bảng không có dòng tiếp nối. LUÔN dùng SELECT DISTINCT * trên TOÀN BỘ CỘT của bảng đã điền
       (như CTE d ở mẫu trên) trước khi SUM/COUNT — nếu không, doanh thu/số lượng bị thổi phồng.
       TUYỆT ĐỐI KHÔNG viết SELECT DISTINCT trên TẬP CON cột (vd. chỉ DISTINCT st, ten_hang, so_luong):
       sẽ gộp nhầm các dòng mua lặp lại hợp lệ (2 ly cà phê cùng đơn, cùng ngày) và làm sai cả SUM lẫn TOP.
       Bảng không có dòng tiếp nối thì chỉ cần: FROM (SELECT DISTINCT * FROM "<ten_bang>" WHERE ...) t
   (c) LOẠI DÒNG TỔNG: nhiều bảng có các dòng "TỔNG NGÀY 03/01", "Tổng tháng 4/2026", "Tổng cộng" với
       ngày/đơn/số lượng NULL — đây là dòng cộng dồn, KHÔNG phải bán hàng. LUÔN thêm ĐỦ CẢ HAI điều kiện
       sau vào WHERE trước khi SUM/COUNT (chỉ lọc tên 'TỔNG' là KHÔNG đủ, vì dòng tổng có thể tên khác):
         <cot_so_luong> IS NOT NULL  AND  vn_norm(<cot_ten_hang>) NOT LIKE 'tong%'
       Nếu tính cả dòng tổng, doanh thu sẽ bị thổi phồng gấp nhiều lần.
   (d) NHIỀU FILE/SHEET: mỗi bảng có thể chỉ phủ một giai đoạn (sheet1 = 2025 + đầu 2026, sheet2 = 2026...),
       có thể chồng lấn vài ngày. Chỉ dùng bảng chứa đúng giai đoạn câu hỏi; nếu buộc ghép nhiều bảng,
       dùng UNION (không phải UNION ALL) trên cùng bộ cột để loại dòng trùng.
   Đếm đơn/khách dùng COUNT(DISTINCT "ma_don"/"ma_kh") trên dữ liệu đã điền. Không khử trùng sẽ bị nhân đôi doanh thu/số lượng.
4. TRẠNG THÁI: mặc định LOẠI đơn hủy/trả/hoàn hàng/đổi/không giao được, nhưng VẪN TÍNH đơn đang giao/đang xử lý (chỉ trừ khi câu hỏi hỏi rõ về hủy/trả):
   vn_norm("trang_thai") NOT LIKE 'huy%' AND vn_norm("trang_thai") NOT LIKE '% huy%'
   AND vn_norm("trang_thai") NOT LIKE '%cancel%'
   AND vn_norm("trang_thai") NOT LIKE '%tra%' AND vn_norm("trang_thai") NOT LIKE '%return%'
   AND vn_norm("trang_thai") NOT LIKE '%hoan hang%' AND vn_norm("trang_thai") NOT LIKE '%hoan tra%'
   AND vn_norm("trang_thai") NOT LIKE '%doi tra%' AND vn_norm("trang_thai") NOT LIKE '%that bai%'
   AND vn_norm("trang_thai") NOT LIKE '%khong giao%'
   TUYỆT ĐỐI KHÔNG dùng NOT LIKE '%huy%' — chuỗi "chuyen" (trong "Đang vận chuyển") CÓ chứa "huy",
   sẽ xóa nhầm đơn đang giao. Cũng KHÔNG dùng '%hoan%' (sẽ xóa nhầm "Đã hoàn thành" = completed).
   Nếu cột trạng thái là TIẾNG ANH (completed/shipping/cancelled/returned...), viết đúng:
   vn_norm("tinh_trang") = 'completed' (hoặc = 'delivered' nếu bảng không có completed).
   Không dùng NOT LIKE cho tiếng Anh vì sẽ vô tình giữ cả 'shipping'. Bảng không có cột trạng thái thì bỏ qua.
5. DOANH THU: nếu bảng có cả cột trước giảm giá và sau giảm giá, "doanh thu"/"tiền hàng" = cột TRƯỚC giảm giá
   ("tong_tien_hang", "tien_hang_truoc_giam", "thanh_tien", "t_tien", "tri_gia", "amount").
   Chỉ dùng cột sau giảm giá khi câu hỏi nói rõ "sau giảm giá"/"thực thu"/"thanh toán".
   Khi có bảng CHI TIẾT bán hàng, tiền hàng/doanh thu LUÔN = SUM(cột thành tiền trên chi tiết đã khử trùng).
   KHÔNG cộng tong_bill/tong_don/"tổng" ở bảng header — header thường LỆCH so với tổng dòng chi tiết.
   "Doanh thu trung bình trên một đơn" = SUM(tiền chi tiết khử trùng) * 1.0 / COUNT(DISTINCT mã_đơn).
   Khi JOIN header với chi tiết, SUM tiền ở bảng chi tiết (hoặc gộp header trước bằng CTE) để tránh nhân dòng.
6. Số: cột TEXT có thể chứa "55.000đ" hoặc "1tr440": CAST(REPLACE(REPLACE(REPLACE("cot", ',', ''), '.', ''), 'đ', '') AS REAL); cột đã INTEGER/REAL thì dùng trực tiếp.
   Không SUM(cột đơn giá). Giá bình quân = SUM(đơn_giá * số_lượng) / NULLIF(SUM(số_lượng), 0).
7. Ngày trong DB là TEXT dạng dd/mm/yyyy hoặc dd/mm/yyyy HH:MM:SS. Dùng các hàm có sẵn:
   - vn_ym("ngay") = '202511'  (lọc tháng 11/2025)
   - vn_ymd("ngay") = '20251107' (lọc đúng ngày) hoặc BETWEEN '20251101' AND '20251130'
   - substr(vn_ymd("ngay"), 1, 4) = '2025' (lọc năm)
   Không dùng DATE_TRUNC/EXTRACT (SQLite không có).
8. TÊN HÀNG/KHÁCH/NCC:
   - Tìm theo từ khóa: vn_like("cot", 'từ khóa') — bỏ dấu, khớp chứa, không phân biệt hoa thường.
     vn_like nhận TỪ KHÓA THUẦN, KHÔNG kèm dấu % (hàm đã tự thêm). Sai: vn_like("hang", '%(m)%');
     đúng: vn_like("hang", '(m)'). Nếu câu nêu ĐƠN VỊ (ly/chai/kg/bao/hộp...) thì lọc thêm
     cột đơn vị: "unit" = 'ly' (hoặc "don_vi_tinh" = 'ly').
   - Hỏi 1 sản phẩm cụ thể kèm size/quy cách (vd. "Nước ép dưa hấu (L)"): dùng đẳng thức đã chuẩn hóa
     vn_norm("ten") = vn_norm('Nước ép dưa hấu (L)') — KHÔNG dùng vn_like vì size nằm trong tên.
   - Nhóm theo tên: GROUP BY vn_norm("ten") để gộp biến thể hoa thường/khoảng trắng; SELECT MIN("ten") làm nhãn.
   - Hỏi theo HOẠT CHẤT (vd. "thuốc có hoạt chất Natri clorid") hoặc "thuốc kê đơn": nối bảng sản phẩm với bảng
     danh mục thuốc theo khóa (ma_sku/ma_hh), rồi lọc bằng SO KHỚP CHÍNH XÁC đã chuẩn hóa:
     vn_norm("hoat_chat") = vn_norm('Natri clorid')  — KHÔNG dùng LIKE, vì sẽ gộp nhầm các hoạt chất phối hợp
     ("Natri clorid + Kali clorid..."). Thuốc kê đơn: "thuoc_ke_don" = '1'. Map sang tên sản phẩm để tính doanh thu
     trên bảng chi tiết bán hàng; lấy TẤT CẢ mã cùng hoạt chất đó.
   - Hỏi theo NHÓM HÀNG (hải sản, thực phẩm chức năng, vitamin...): dùng cột phân loại/nhóm trong bảng danh mục
     (vn_like) hoặc tên nhóm; kiểm tra schema trước khi suy đoán từ tên món.
9. CA LÀM VIỆC (bảng có cột giờ/thời gian): ca sáng = 06:00–13:59; ca chiều = 14:00–21:59.
   Lấy giờ: CAST(substr("thoi_gian_tao",12,2) AS INTEGER) BETWEEN 6 AND 13 (sáng) hoặc 14 AND 21 (chiều).
10. JOIN: so sánh TRỰC TIẾP cột = cột (vd. ct."ma_giao_dich" = h."order_id").
    TUYỆT ĐỐI KHÔNG bọc CAST quanh cột ở điều kiện JOIN — CAST làm mất index, SQLite chạy O(n²) và bị timeout.
    Chỉ khi thật sự lệch kiểu mới CAST (ưu tiên CAST vế của bảng nhỏ/subquery, không CAST cột bảng lớn).
    CHIA: trong SQLite, số nguyên / số nguyên = phép chia NGUYÊN (mất phần lẻ). Luôn nhân 1.0:
    SUM(x) * 1.0 / NULLIF(SUM(y), 0) — nếu không, giá bình quân sẽ lệch 1 đơn vị.
    Phần trăm: * 100.0 và ROUND(..., 4) — KHÔNG ROUND 2 (dung sai chấm là 0.01 nên làm tròn 2 dễ trượt).
11. Top/N/"bán chạy nhất": ORDER BY số lượng đã quy đổi DESC LIMIT N — ưu tiên cột "so_luong_le"/"so_luong_quy_doi";
    loại dòng phí dịch vụ (vn_norm("ten") NOT LIKE '%phi phuc vu%'). Hỏi doanh thu thì ORDER BY SUM(tiền).
12. Đơn vị tính: dùng hệ số quy đổi trong README hoặc bảng DanhMuc/QuyDoi; quy về đơn vị cơ bản khi tổng hợp/so sánh số lượng.
13. NHIỀU GIÁ TRỊ THEO CHUỖI ("mỗi tháng", "mỗi loại", "lần lượt", "A và B mỗi loại bao nhiêu",
    "mấy X chẵn, mấy Y lẻ") → type=list: SELECT ĐÚNG 1 CỘT giá trị, N dòng, thứ tự thời gian/nhóm tăng dần.
    TUYỆT ĐỐI KHÔNG kèm cột tháng/nhãn/phân loại — đáp án gold là mảng phẳng các số, kèm nhãn sẽ bị chấm SAI.
    Không trả nhiều cột: dùng SUM(CASE WHEN nhóm='X' THEN ... END) cho từng cột rồi SELECT các cột đó.
    CHỈ dùng type=table khi câu hỏi yêu cầu kèm TÊN đối tượng, vd "3 khách nợ nhiều nhất kèm số nợ"
    → columns=[khach_hang, so_tien_no], rows=[[tên, số], ...].
14. "Cùng kỳ tháng trước" = cùng số ngày tính từ đầu tháng trước. "Tăng/giảm bao nhiêu %" = (cur - prev) / NULLIF(prev, 0) * 100, ROUND(...,4).
15. NGƯỠNG THEO ĐƠN/KHÁCH ("đơn >= X", "khách mua >= N đơn"): GROUP BY đơn/khách rồi HAVING SUM(tiền) >= X (hoặc HAVING COUNT(DISTINCT đơn) >= N), đếm bằng COUNT(*).
    KHÁCH LẺ VÃNG LAI: các giá trị khách như 'Khách lẻ', 'lẻ', 'kl', 'khách lẻ' là khách ẩn danh (không phải 1 người).
    LOẠI chúng khỏi mọi thống kê THEO KHÁCH (đếm khách, tỉ lệ % khách, top khách, số đơn/khách):
      vn_norm("khach_hang") NOT IN ('khach le', 'le', 'kl')
    nhưng VẪN tính chúng khi thống kê doanh thu/đơn hàng nói chung (trừ khi câu hỏi nói rõ "khách lẻ").
    Công nợ: chỉ tính đơn ĐÃ GIAO/hoàn tất (không tính đang vận chuyển, hủy, trả).
    Đếm khách còn nợ: GROUP BY khách, HAVING SUM("con_lai"/"con_no") > 0. Tổng nợ: SUM("con_no") của các đơn đó.
15b. TỈ LỆ ĐƠN HỦY = số đơn có trạng thái hủy (vn_norm LIKE 'huy%' OR vn_norm LIKE '% huy%') / TỔNG SỐ ĐƠN (mẫu số KHÔNG lọc trạng thái) * 100.
    Đếm đơn theo COUNT(DISTINCT "ma_don_hang") trên dữ liệu đã điền trạng thái; đơn "hoàn hàng" KHÔNG tính là hủy.
15c. "Giá nhập/bán bình quân 1 đơn vị" = SUM(thanh_tien) * 1.0 / NULLIF(SUM(số_lượng quy đổi), 0) — tính trên dòng chi tiết đã khử trùng.
15d. "Số đợt/chuyến giao hàng" = COUNT(DISTINCT mã_đơn || '-' || lần_giao) trong bảng giao hàng (mỗi đơn có thể giao nhiều đợt).
16. HÌNH DẠNG ĐÁP ÁN: number = 1 dòng 1 cột; percent = 1 số đã *100; list = ĐÚNG 1 CỘT N dòng giá trị
    (KHÔNG kèm cột nhãn/tháng — kể cả câu "mỗi tháng"/"mỗi loại"); table = nhiều cột có TÊN đối tượng;
    date = 1 ngày; string = 1 chuỗi; boolean = 1/0. Không trả cột thừa.
17. Nếu thông tin cần thiết KHÔNG có trong schema (vd. địa chỉ/đường/phố, số điện thoại khách, giới tính khách) → sql=null, type=unanswerable.
18. XỬ LÝ NHIỀU BẢNG CHI TIẾT (File bị tách làm nhiều phần, vd: chitiet_1, chitiet_2): TUYỆT ĐỐI KHÔNG dùng lệnh UNION/UNION ALL trước khi xử lý rowid, vì UNION sẽ làm MẤT thuộc tính rowid trong SQLite, khiến lệnh JOIN của forward-fill bị lỗi 'no such column: s.oid'. CÁCH LÀM ĐÚNG: Áp dụng quy trình forward-fill (CTE s, g, f, d) CHO TỪNG BẢNG RIÊNG BIỆT trước. Sau khi đã xử lý xong từng bảng (đã có f_oid, f_ngay...), mới dùng UNION ALL để gộp các kết quả cuối cùng lại với nhau.
19. ALIAS KHI NHÂN BẢN QUY TRÌNH: Với bảng chi tiết thứ 2 phải đổi ĐỒNG BỘ alias (bảng 1: s1/g1/f1/d1; bảng 2: s2/g2/f2/d2). TUYỆT ĐỐI KHÔNG để sót "s.oid"/"s.ngay" khi không có CTE tên s — đây chính là lỗi 'no such column: s.oid'.
20. BẢNG CHI TIẾT KHÔNG CÓ CỘT NGÀY/TRẠNG THÁI: Kiểm tra schema trước khi viết CTE. Nếu bảng chi tiết KHÔNG có cột ngày/trạng thái (vd. biz_106: bảng chitiet chỉ có "ma_hoa_don", không có "thoi_gian"/"trang_thai"), thì CTE s CHỈ chọn các cột THẬT SỰ tồn tại (thường chỉ "ma_hoa_don"). TUYỆT ĐỐI KHÔNG SELECT cột không tồn tại bằng ngoặc kép — SQLite sẽ coi đó là chuỗi literal, không báo lỗi, khiến lọc ngày/trạng thái trả 0 dòng/NULL. Sau khi forward-fill mã đơn và gộp các bảng chi tiết, JOIN với bảng header theo mã đơn để lấy ngày/trạng thái, rồi lọc trên cột header (vd. h."thoi_gian", h."trang_thai").

LOẠI ĐÁP ÁN (type)
- number: một giá trị số (tiền, số lượng, số đếm).
- percent: một số phần trăm đã nhân 100 (giữ 4 chữ số thập phân, không làm tròn 2).
- list: một cột, nhiều dòng (top N, liệt kê, mỗi tháng một giá trị).
- table: nhiều cột, nhiều dòng (ví dụ mỗi khách kèm số tiền).
- date: một ngày.
- string: một chuỗi (tên hàng, tên khách...).
- boolean: đúng/sai (SQL trả 1/0).
- unanswerable: không trả lời được.
"""

# ==========================================
# 5. HÀM XỬ LÝ DATABASE
# ==========================================
def build_sqlite_db(biz_id):
    biz_path = os.path.join(INPUT_DIR, "data", biz_id)
    # Lưu file database tạm vào /tmp để tránh lỗi Permission Denied
    db_path = f"/tmp/_temp_{biz_id}.db" 
    
    if os.path.exists(db_path):
        try: os.remove(db_path)
        except: pass
    
    conn = sqlite3.connect(db_path)
    conn.create_function("vn_norm", 1, vn_norm)
    conn.create_function("vn_ym", 1, vn_ym)
    conn.create_function("vn_ymd", 1, vn_ymd)
    
    if os.path.exists(biz_path):
        for filename in os.listdir(biz_path):
            filepath = os.path.join(biz_path, filename)
            if filename.endswith(".csv"):
                try:
                    df = pd.read_csv(filepath)
                    df.to_sql(filename.replace(".csv", "").lower(), conn, if_exists="replace", index=False)
                except: pass
            elif filename.endswith(".xlsx"):
                try:
                    xls = pd.ExcelFile(filepath)
                    for sheet in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet)
                        df.to_sql(f"{filename.replace('.xlsx', '')}_{sheet}".lower(), conn, if_exists="replace", index=False)
                except: pass
    conn.close()
    return db_path

def get_schema_and_readme(biz_id, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    schema_str = ""
    for table in cursor.fetchall():
        t_name = table[0]
        cursor.execute(f'PRAGMA table_info("{t_name}");')
        schema_str += f"Table: {t_name} | Columns: {', '.join([c[1] for c in cursor.fetchall()])}\n"
    conn.close()
    
    readme_path = os.path.join(INPUT_DIR, "data", biz_id, "README.md")
    readme_str = open(readme_path, "r", encoding="utf-8").read() if os.path.exists(readme_path) else ""
    return schema_str, readme_str

# ==========================================
# 6. VÒNG LẶP CHẠY INFERENCE
# ==========================================
questions_path = os.path.join(INPUT_DIR, "questions.jsonl")
with open(questions_path, "r", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

results = []
print(f"🚀 Bắt đầu chạy {len(questions)} câu hỏi...")

for idx, q in enumerate(questions):
    qid, biz_id, as_of_date = q["id"], q["business_id"], q.get("as_of_date", "")
    db_path = build_sqlite_db(biz_id)
    schema, readme = get_schema_and_readme(biz_id, db_path)
    
    user_prompt = f"Doanh nghiệp: {biz_id}\nREADME:\n{readme}\nSchema:\n{schema}\nHôm nay là: {as_of_date}\nCâu hỏi: {q['question']}\nTrả về JSON."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt", 
        max_length=4096, 
        truncation=True
    ).to(model.device)
    
    outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True)
    response_text = tokenizer.batch_decode(outputs)[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        ai_output = json.loads(json_match.group(0)) if json_match else {}
        
        sql = ai_output.get("sql")
        ans_type = ai_output.get("type", "number")
        
        if sql is None or ans_type == "unanswerable":
            pred = {"value": None}
        else:
            conn = sqlite3.connect(db_path)
            conn.create_function("vn_norm", 1, vn_norm)
            conn.create_function("vn_ym", 1, vn_ym)
            conn.create_function("vn_ymd", 1, vn_ymd)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            if df.empty:
                pred = {"value": None}
            else:
                if ans_type == "list": 
                    pred = {"items": df.iloc[:, 0].tolist()}
                elif ans_type == "table": 
                    pred = {"columns": df.columns.tolist(), "rows": df.values.tolist()}
                else:
                    val = df.iloc[0, 0]
                    if isinstance(val, (int, float)): 
                        val = float(val) if '.' in str(val) else int(val)
                    pred = {"value": val}
    except Exception:
        ans_type = "unanswerable"
        pred = {"value": None}

    results.append({"id": qid, "type": ans_type, "answer": pred})
    
    if (idx + 1) % 50 == 0:
        print(f"  ✅ Đã xong {idx + 1}/{len(questions)}")

# Ghi file kết quả vào thư mục /result của BTC
output_path = os.path.join(OUTPUT_DIR, "submission.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for res in results:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")

print(f"🎉 HOÀN THÀNH! File kết quả đã lưu tại: {output_path}")