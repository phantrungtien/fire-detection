# phantrungtien-fire-detection

Fire & smoke detection, YOLO11. Gói này chứa model tốt nhất, code chạy, và kết quả train.

Classes: `0 = Fire`, `1 = Smoke`

```
models/     best_nano_111.pt + 2 bản ONNX
code/       detect.py + fire_detector.py + test_detect.py
training/   notebook, biểu đồ train, bảng so sánh model
samples/    ảnh/video để thử
```

## Model nào tốt nhất

`best_nano_111.pt` — cao nhất trên cả 6 model đã train.

| Model | fitness | mAP50 | mAP50-95 | P | R | Size |
|---|---|---|---|---|---|---|
| **best_nano_111** | **0.5198** | 0.7694 | **0.4921** | 0.8068 | 0.7172 | 5.5 MB |
| best_nano | 0.5173 | 0.7768 | 0.4884 | 0.8015 | 0.7304 | 5.5 MB |
| best_small | 0.5102 | 0.7713 | 0.4812 | 0.7721 | 0.7373 | 19.2 MB |
| yolo11-d-fire-dataset | 0.5026 | 0.7856 | 0.4712 | 0.7841 | 0.7237 | 5.5 MB |
| best_medium | 0.4559 | 0.7202 | 0.4265 | 0.7826 | 0.6360 | 40.5 MB |
| best_large | 0.4456 | 0.6865 | 0.4188 | 0.7730 | 0.6247 | 51.2 MB |

Số liệu đọc trực tiếp từ `train_metrics` nhúng trong mỗi checkpoint, không phải chạy lại val. Chi tiết: `training/model_comparison.csv`.

Điểm đáng chú ý: **model to hơn lại tệ hơn**. Large kém nano ~17% mAP50-95 dù nặng gấp 10 lần — dấu hiệu overfit trên tập 10k ảnh. Nên nano vừa chính xác nhất vừa nhẹ nhất, rất hợp Raspberry Pi 5.

`yolo11-d-fire-dataset.pt` train trên dataset khác (D-Fire) và **đảo thứ tự class** (`0=smoke, 1=fire`). Đừng thay vào code hiện tại nếu không sửa mapping.

## File model

| File | Dùng khi |
|---|---|
| `best_nano_111.pt` | PC có GPU |
| `best_nano_111_640.onnx` | Pi 5, ưu tiên chính xác |
| `best_nano_111_320.onnx` | Pi 5, ưu tiên tốc độ (~3x nhanh hơn) |

ONNX: opset 12, đã simplify, `imgsz` nằm trong metadata nên không cần truyền lúc infer.

Đo trên CPU x86 (Pi 5 sẽ chậm hơn, cần đo lại trên máy thật):
`640 → 54 ms/frame`, `320 → 18 ms/frame`.

Bản 320 đổi tốc độ lấy độ nhạy: trên `samples/house.png` confidence smoke tụt 0.63 → 0.45. Nếu dùng 320 thì hạ `--conf` xuống ~0.2.

## Khởi tạo

Chạy mọi lệnh dưới đây từ thư mục gốc của gói này.

```bash
cd phantrungtien-fire-detection

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install ultralytics onnxruntime opencv-python cvzone
```

Trên Raspberry Pi 5 dùng `opencv-python-headless` thay cho `opencv-python` (nhẹ hơn, không cần GUI), và luôn chạy kèm `--no-show`.

Kiểm tra cài đặt xong chưa:

```bash
python -c "import ultralytics, onnxruntime, cv2, cvzone; print('OK', ultralytics.__version__)"
```

## Chạy

```bash
python code/detect.py samples/house.png
python code/detect.py samples/gen_fire.mp4 --save
python code/detect.py 0                                 # webcam
python code/detect.py samples/ --no-show --save         # cả thư mục
python code/detect.py 0 --model models/best_nano_111_320.onnx --no-show
```

Tuỳ chọn: `--conf` (mặc định 0.35), `--iou`, `--smoke-conf`, `--save`, `--no-show` (cho Pi headless).
Kết quả ghi vào `outputs/`. Dừng bằng `q` (có cửa sổ) hoặc Ctrl+C.

## Test các đoạn code

### 1. Self-check (nhanh, không cần model chạy)

```bash
python code/test_detect.py
```

Kiểm tra phần định tuyến nguồn vào (webcam / ảnh / video / thư mục / file không tồn tại) và sự tồn tại của 3 file model. Kết quả mong đợi — 6 dòng `ok` rồi `tất cả check đã pass`:

```
ok  test_folder_yields_images_only
ok  test_image_file_is_not_a_stream
ok  test_missing_source_exits
ok  test_models_present
ok  test_video_file_is_a_stream
ok  test_webcam_index_is_a_stream
tất cả check đã pass
```

Chạy được cả bằng pytest (`pytest code/test_detect.py`) nếu môi trường có pytest lành lặn.

### 2. Test 1 ảnh

```bash
python code/detect.py samples/house.png --no-show
```

Mong đợi: `house: Fire`. Thêm `--save` thì có `outputs/house.jpg` với box đỏ `Fire` + box xám `Smoke`.

### 3. Test cả thư mục

```bash
python code/detect.py samples/ --no-show
```

Mong đợi đúng 5 ảnh (video `.mp4` bị bỏ qua, đúng thiết kế):

```
ex1: Fire
ex2: Fire
ex3: Smoke
ex4: Fire
house: Fire
```

### 4. Test video

```bash
python code/detect.py samples/gen_fire.mp4 --no-show --save
```

Mong đợi `gen_fire: 224/253 frames with fire or smoke`, và `outputs/gen_fire.mp4` mở được.

(Con số phụ thuộc model: `.pt` cho 224/253, ONNX 320 cho 235/253 — bản 320 nhạy hơn ở ngưỡng mặc định nhưng confidence thấp hơn.)

Kiểm tra file xuất có hợp lệ không:

```bash
python -c "import cv2; c=cv2.VideoCapture('outputs/gen_fire.mp4'); print('đọc được:', c.read()[0], '| frames:', int(c.get(cv2.CAP_PROP_FRAME_COUNT)))"
```

### 5. Test 3 model cho ra kết quả khớp nhau

```bash
python -c "
import sys; sys.path.insert(0,'code')
import cv2
from fire_detector import Detector
frame = cv2.imread('samples/house.png')
for m in ['models/best_nano_111.pt','models/best_nano_111_640.onnx','models/best_nano_111_320.onnx']:
    print(m, '->', Detector(m, min_confidence=0.35, iou_threshold=0.1).process_frame(frame.copy())[1])
"
```

Cả 3 phải ra `Fire`. Confidence tham chiếu (đo trên CPU x86): `.pt` 0.835, ONNX 640 0.841, ONNX 320 0.793.

### 6. Test webcam

```bash
python code/detect.py 0 --model models/best_nano_111_320.onnx --no-show
```

Hướng camera vào chỗ bình thường thì không được báo gì (không false positive). Dừng bằng Ctrl+C.

Nếu camera không mở được, kiểm tra riêng:

```bash
python -c "import cv2; c=cv2.VideoCapture(0); print('mở được:', c.isOpened(), '| đọc được:', c.read()[0])"
```

### Mức đã kiểm chứng

| Hạng mục | Trạng thái |
|---|---|
| Self-check `test_detect.py` | Đã chạy trong gói này, 6/6 pass |
| Ảnh, thư mục, video, webcam, 3 model | Đã chạy trên repo gốc, cùng file nguồn |
| Ctrl+C lúc `--save` webcam | **Chưa kiểm chứng** — xem mục dưới |

Đoạn xử lý tín hiệu dừng (`SIGINT`/`SIGTERM`) mới thêm sau lần chạy cuối nên chưa được test thực tế. Lần đầu quay webcam có `--save`, nhớ thử Ctrl+C rồi mở lại file `outputs/webcam0.mp4` xem đọc được không. Nếu báo `moov atom not found` nghĩa là writer chưa đóng đúng, cần sửa lại.

Lưu ý: đừng dùng `kill -9` để dừng lúc đang `--save` — file mp4 sẽ hỏng hoàn toàn vì thiếu moov atom.

## Kết quả train

- `training/smoke-fire-detection-yolo11.ipynb` — notebook train
- `training/training_graphs.png` — biểu đồ loss/mAP
- `training/model_comparison.csv` — bảng so sánh 6 model

Cấu hình train (từ notebook): `yolo11n.pt`, 250 epochs, imgsz 640, batch 32, patience 20. Dừng sớm do 20 epoch không cải thiện, tốt nhất ở epoch 92.

Theo class, model tốt nhất:

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Fire | 0.813 | 0.806 | 0.828 | 0.513 |
| Smoke | 0.800 | 0.629 | 0.711 | 0.472 |

Smoke recall chỉ 0.63 — đây là điểm yếu rõ nhất. Khói bị bỏ sót nhiều hơn lửa, nếu cảnh cần bắt khói sớm thì hạ `--conf`.

## Dữ liệu train — KHÔNG có trong gói này

Dataset 10.463 ảnh không nằm trong repo gốc, nó ở trên Roboflow và được tải lúc train. `samples/` chỉ là media demo, không phải tập train.

Tải về:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("sayed-gamall").project("fire-smoke-detection-yolov11")
dataset = project.version(2).download("yolov11")
```

| Split | Images | Annotations |
|---|---|---|
| Train | 9.156 | 27.468 |
| Val | 872 | 2.616 |
| Test | 435 | 1.305 |

Cần API key Roboflow. Tải xong sẽ có `data.yaml` để train hoặc chạy `yolo mode=val`.

## Lưu ý khi deploy lên Pi

`detect.py` set sẵn `YOLO_AUTOINSTALL=false`. Không bỏ dòng này — nếu không ultralytics sẽ thử `pip install` mỗi lần load ONNX, tốn ~60s/lần và cần mạng.

Trên Pi nên thử thêm format NCNN, thường nhanh hơn ONNX 2-3x trên CPU ARM:

```python
from ultralytics import YOLO
YOLO("models/best_nano_111.pt").export(format="ncnn", imgsz=320)
```
