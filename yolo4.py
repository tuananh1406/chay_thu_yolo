#! coding: utf-8
import numpy as np
import argparse
import time
import cv2
import os


tuy_chon = argparse.ArgumentParser()
tuy_chon.add_argument(
        "-i",
        "--image",
        required=True,
        help="Đường dẫn tệp ảnh",
    )
tuy_chon.add_argument(
        "-y",
        "--yolo",
        default='./',
        help="Đường dẫn đến thư mục yolo",
    )
tuy_chon.add_argument(
        "-c",
        "--dochinhxac",
        type=float,
        default=0.5,
        help="Độ chính xác tối thiểu",
    )
tuy_chon.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="Ngưỡng để sử dụng cho thuật toán non-maxima suppression",
    )
cac_tuy_chon = vars(tuy_chon.parse_args())

# Mở tệp tên nhãn của COCO
print('[INFO] Tải danh sách tên đối tượng')
duong_dan_nhan = os.path.sep.join([
        cac_tuy_chon['yolo'],
        'coco.names',
    ])
with open(duong_dan_nhan) as tep_tin:
    DS_NHAN = tep_tin.read().strip().split("\n")

# Tạo danh sách màu ngẫu nhiên cho các lớp phát hiện được
print('[INFO] Tạo danh sách màu ngẫu nhiên cho từng nhãn')
np.random.seed(42)
DS_MAU = np.random.randint(
        0,
        255,
        size=(len(DS_NHAN), 3),
        dtype='uint8',
    )

# Tải mô hình YOLO đã huấn luyện và tệp cấu hình
duong_dan_weights = os.path.sep.join([
    cac_tuy_chon["yolo"],
    "yolov4.weights",
    ])
duong_dan_cau_hinh = os.path.sep.join([
    cac_tuy_chon["yolo"],
    "yolov4.cfg",
    ])
print("[INFO] tải mô hình YOLO...")
mo_hinh = cv2.dnn.readNetFromDarknet(
        duong_dan_cau_hinh,
        duong_dan_weights,
        )

# Tải ảnh đầu vào và lấy kích thước
print('[INFO] Tải hình ảnh')
hinh_anh = cv2.imread(cac_tuy_chon["image"])
(H, W) = hinh_anh.shape[:2]

# Lấy đầu ra các tên lớp từ YOLO
ds_ten_lop = mo_hinh.getLayerNames()
ds_ten_lop = [ds_ten_lop[i[0] - 1] for i in mo_hinh.getUnconnectedOutLayers()]

# Tạo blob từ ảnh đầu vào và đưa vào mô hình để xác định đối tượng trong ảnh
blob = cv2.dnn.blobFromImage(
        hinh_anh,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False,
    )
mo_hinh.setInput(blob)
start = time.time()
lop_ket_qua = mo_hinh.forward(ds_ten_lop)
thoi_gian_xu_ly = time.time() - start

# Hiển thị thời gian xử lý
print("[INFO] Thời gian xác định đối tượng: {:.6f} giây".format(
    thoi_gian_xu_ly,
    ))

# Khởi tạo các danh sách chứa thông tin đầu ra
ds_duong_bao = []
ds_do_chinh_xac = []
ds_id_lop = []

# Lọc qua từng lớp kết quả
print('[INFO] Vẽ đường bao xung quanh các đối tượng')
for ket_qua in lop_ket_qua:
    # Lặp qua từng đối tượng phát hiện
    for doi_tuong in ket_qua:
        # Lấy id lớp và độ chính xác
        cac_chi_so = doi_tuong[5:]
        id_lop = np.argmax(cac_chi_so)
        do_chinh_xac = cac_chi_so[id_lop]

        # Lọc các kết quả có độ chính xác thấp
        if do_chinh_xac > cac_tuy_chon['dochinhxac']:
            # Tính lại kích thước đường bao theo đúng kích thước
            #  của ảnh đầu vào, YOLO chỉ trả về tọa độ tâm (x, y) và chiều dài,
            # rộng của đường bao
            # Tham khảo thêm:
            # https://howtothink.readthedocs.io/en/latest/PvL_06.html
            duong_bao = doi_tuong[0:4] * np.array([W, H, W, H])
            (x_tam, y_tam, chieu_dai, chieu_rong) = duong_bao.astype("int")
            # Sử dụng tọa độ tâm x, y để lấy góc trên bên trái của đường bao
            x = int(x_tam - (chieu_dai / 2))
            y = int(y_tam - (chieu_rong / 2))

            # Lưu kết quả vào các danh sách tương ứng
            ds_duong_bao.append([x, y, int(chieu_dai), int(chieu_rong)])
            ds_do_chinh_xac.append(float(do_chinh_xac))
            ds_id_lop.append(id_lop)

# Áp dụng thuật toán non-maxima suppression để gộp các đường bao
# chồng lên nhau thành 1
print('[INFO] Gộp các đường bao có chung đối tượng')
idxs = cv2.dnn.NMSBoxes(
        ds_duong_bao,
        ds_do_chinh_xac,
        cac_tuy_chon['dochinhxac'],
        cac_tuy_chon['threshold'],
    )
if len(idxs) > 0:
    for duong_bao in idxs.flatten():
        # Lấy tọa độ đường bao
        (x, y) = (ds_duong_bao[duong_bao][0], ds_duong_bao[duong_bao][1])
        (w, h) = (ds_duong_bao[duong_bao][2], ds_duong_bao[duong_bao][3])

        # Vẽ đường bao quanh đối tượng và hiển thị nhãn trên ảnh
        mau_sac = [int(c) for c in DS_MAU[ds_id_lop[duong_bao]]]
        cv2.rectangle(
                hinh_anh,
                (x, y),
                (x + w, y + h),
                mau_sac,
                2,
            )
        nhan = "{}: {:.4f}".format(
                DS_NHAN[ds_id_lop[duong_bao]],
                ds_do_chinh_xac[duong_bao],
                )
        cv2.putText(
                hinh_anh,
                nhan,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                mau_sac,
                2,
            )

# Hiển thị ảnh kết quả
print('[INFO] Hiển thị kết quả')
if W > 1000 or H > 1000:
    hinh_anh_moi = cv2.resize(hinh_anh, (int(W / 2), int(H / 2)))
else:
    hinh_anh_moi = hinh_anh
cv2.imshow("Ket qua", hinh_anh_moi)
cv2.waitKey(0)
