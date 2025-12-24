from paddleocr import PaddleOCR
import inspect

try:
    print("Inspect PaddleOCR init:")
    print(inspect.signature(PaddleOCR.__init__))
except Exception as e:
    print(e)

try:
    print("\nDocstring:")
    print(PaddleOCR.__doc__)
except Exception as e:
    print(e)
