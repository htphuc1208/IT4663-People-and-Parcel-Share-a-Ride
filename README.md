## Neighbourhood Operators

Các toán tử lân cận được sử dụng trong Tabu Search nhằm cân bằng giữa việc **tái phân bổ yêu cầu giữa các taxi** và **tối ưu thứ tự phục vụ bên trong từng tuyến**.

### 1. `parcel_transfer`

Chuyển toàn bộ cặp **pickup/drop-off** của một bưu kiện từ taxi hiện tại sang một taxi khác.

**Mục đích:**
- Cân bằng tải giữa các taxi.
- Giảm chiều dài của tuyến dài nhất.
- Phù hợp trực tiếp với mục tiêu **Min-Max** của bài toán.

---

### 2. `parcel_swap`

Hoán đổi vị trí của hai bưu kiện trong lời giải.

**Mục đích:**
- Cải thiện thứ tự phục vụ bưu kiện.
- Giảm các đoạn đường vòng (detour) không cần thiết.
- Khai thác các cơ hội tối ưu cục bộ mà không thay đổi phân công taxi.

---

### 3. `passenger_relocate`

Di chuyển một hành khách sang vị trí hoặc taxi khác.

**Mục đích:**
- Điều chỉnh phân công hành khách giữa các tuyến.
- Tăng tính linh hoạt của quá trình tìm kiếm.

**Lưu ý:**
Trong mô hình giải mã hiện tại, mỗi hành khách được phục vụ theo dạng chuyến trực tiếp:

```text
pickup → drop-off
```

Do đó toán tử chỉ cần di chuyển điểm **pickup**, còn điểm **drop-off** sẽ được sinh tự động khi giải mã.

---

### 4. `intra_route_reorder`

Đảo ngược một đoạn trong cùng một tuyến (local 2-opt).

**Mục đích:**
- Tối ưu thứ tự các điểm thăm trong một taxi.
- Giảm quãng đường di chuyển bên trong tuyến.
- Hiệu quả khi việc phân công khách/bưu kiện cho taxi đã tương đối ổn định.

**Ví dụ:**

```text
A → B → C → D → E
```

Sau khi áp dụng:

```text
A → D → C → B → E
```

---

### Summary

| Operator | Scope | Main Purpose |
|-----------|--------|--------------|
| `parcel_transfer` | Inter-route | Balance workload and reduce the longest route |
| `parcel_swap` | Inter/Intra-route | Improve parcel service ordering |
| `passenger_relocate` | Inter/Intra-route | Reassign passenger requests |
| `intra_route_reorder` | Intra-route | Local route optimization (2-opt style) |