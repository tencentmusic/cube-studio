import base64
import io
from paddleocr import PaddleOCR, draw_ocr
from PIL import ImageGrab, Image
import numpy


def s_ocr():
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    return ocr


def ocr_test(ocr, np_img):
    text = ''
    result = ocr.ocr(np_img, cls=True)  # cls：测试是否需要旋转180°，影响性能，90°以及270°，无需开启。
    for one in result:
        text += one[1][0] + '\r\n'

    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(np_img, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    base64_str = img_base64(im_show)
    return base64_str, text[:-2]


def img_base64(img, coding='utf-8'):
    img_format = img.format
    if img_format is None:
        img_format = 'JPEG'

    format_str = 'JPEG'
    if 'png' == img_format.lower():
        format_str = 'PNG'
    if 'gif' == img_format.lower():
        format_str = 'gif'

    if img.mode == "P":
        img = img.convert('RGB')
    if img.mode == "RGBA":
        format_str = 'PNG'
        img_format = 'PNG'

    output_buffer = io.BytesIO()
    # img.save(output_buffer, format=format_str)
    img.save(output_buffer, quality=100, format=format_str)
    byte_data = output_buffer.getvalue()
    base64_str = 'data:image/' + img_format.lower() + ';base64,' + base64.b64encode(byte_data).decode(coding)

    return base64_str


if __name__ == '__main__':
    ocr = s_ocr()  # 实例化ocr

    # 本地测试使用截图进行测试（环境：Windows10）
    image = ImageGrab.grab()  # 截图
    np_image = numpy.array(image)  # 将图片转为ndarray

    # 正式环境使用base64用此方法
    # image_decode = base64.b64decode(data['image_data'])
    # nparr = np.fromstring(image_decode, np.uint8)

    image_base, text = ocr_test(ocr, np_image)
    print(len(image_base))
    print(text)
