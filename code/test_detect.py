"""Self-check cho detect.py — không cần pytest.

    python code/test_detect.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLES = ROOT / "samples"

sys.path.insert(0, str(HERE))
from detect import resolve_sources


def test_webcam_index_is_a_stream():
    assert list(resolve_sources("0")) == [("webcam0", 0, True)]


def test_video_file_is_a_stream():
    label, _, is_stream = next(resolve_sources(str(SAMPLES / "gen_fire.mp4")))
    assert (label, is_stream) == ("gen_fire", True)


def test_image_file_is_not_a_stream():
    label, _, is_stream = next(resolve_sources(str(SAMPLES / "house.png")))
    assert (label, is_stream) == ("house", False)


def test_folder_yields_images_only():
    found = list(resolve_sources(str(SAMPLES)))
    assert not any(is_stream for _, _, is_stream in found), "video lọt vào danh sách ảnh"
    labels = sorted(label for label, _, _ in found)
    assert labels == ["ex1", "ex2", "ex3", "ex4", "house"], labels


def test_missing_source_exits():
    try:
        list(resolve_sources(str(SAMPLES / "khong_ton_tai.png")))
    except SystemExit:
        return
    raise AssertionError("source không tồn tại phải SystemExit")


def test_models_present():
    for name in ("best_nano_111.pt", "best_nano_111_640.onnx", "best_nano_111_320.onnx"):
        assert (ROOT / "models" / name).exists(), f"thiếu model {name}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("tất cả check đã pass")
