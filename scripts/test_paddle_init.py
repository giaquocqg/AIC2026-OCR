import paddleocr
from paddleocr import PaddleOCR
import paddle

print("Paddle version:", paddleocr.__version__)
print("Paddle CUDA available:", paddle.is_compiled_with_cuda())

try:
    ocr = PaddleOCR(use_angle_cls=True, lang="vi", use_gpu=paddle.is_compiled_with_cuda())
    print("Init 1 (use_angle_cls, use_gpu) OK")
except Exception as e:
    print("Init 1 Failed:", e)

try:
    ocr = PaddleOCR(lang="vi")
    print("Init 2 (lang='vi') OK")
except Exception as e:
    print("Init 2 Failed:", e)

try:
    ocr = PaddleOCR(use_textline_orientation=True, lang="vi")
    print("Init 3 (use_textline_orientation) OK")
except Exception as e:
    print("Init 3 Failed:", e)
