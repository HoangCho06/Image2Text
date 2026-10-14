import torch, easyocr, pymupdf, os
from rapidocr import RapidOCR
from pathlib import Path

# version 0.2.0

def check_gpu():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name()
    cuda_version = torch.version.cuda
    if gpu_available:
        print(f'gpu: {gpu_name}\nversion: {cuda_version}')
        return True
    else:
        print('No gpu detected')
        return False

def RapidOCR_i2t(engine, file_path, file_name, output_file_path, test=False):
    output_doc = pymupdf.open()
    page_number = -1
    x1, y1 = 10, 20
    black = (0,0,0)
    clean_name = f'{Path(file_name).stem}_result.pdf'

    pix = pymupdf.Pixmap(file_path)
    width = pix.width
    height = pix.height

    result = engine(file_path)
    page = output_doc.new_page(page_number, width, height)
    page.insert_text((x1,y1), result.txts, fontsize=11, color=black)
    page = output_doc.new_page(page_number, width, height)
    
    # result.boxes = [box1, box2, ...]
    # box = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    number_of_text_boxes = len(result.boxes)
    for box_index in range(number_of_text_boxes):
        top_left = (box_index, 0)
        x,y = result.boxes[top_left]

        text = result.txts[box_index]
        page.insert_text((x,y), text, fontsize=11, color=black)
        output_doc.save(f"{output_file_path}/{clean_name}")

    if test:
        result.vis('test.png')
        print(result.txts)
    output_doc.close()

def main():
    input_dir_path = 'image_to_text/input_files'
    input_dir = os.listdir(input_dir_path)
    output_dir_path = 'image_to_text/output_files'

    engine = RapidOCR()
    for file_name in input_dir:
        path_to_file = f"{input_dir_path}/{file_name}"

        try:
            if file_name.lower() != '.pdf':
                RapidOCR_i2t(engine, path_to_file, file_name, output_dir_path)

            else:
                print('PDFs are not supported')
        except Exception as e:
            print(e)
            pass

main()
