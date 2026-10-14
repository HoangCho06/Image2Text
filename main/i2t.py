import torch, pymupdf, os
from rapidocr import RapidOCR

# version 0.1.0

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


def main():
    engine = RapidOCR()
    folder_name = 'main/input_files'
    input_folder = os.listdir(folder_name)

    for file_name in input_folder:
        path_to_file = f"{folder_name}/{file_name}"

        try:
            output_doc = pymupdf.open()
            
            pix = pymupdf.Pixmap(path_to_file)
            width = pix.width
            height = pix.height
            
            result = engine(path_to_file)
            page = output_doc.new_page(-1, width, height)
            number_of_text_boxes = len(result.boxes)
            for text_box_index in range(number_of_text_boxes):
                x,y = result.boxes[text_box_index][0]
                text = result.txts[text_box_index]
                page.insert_text((x,y), text, fontsize=11, color=(0,0,0))
            output_doc.save(f'{file_name}_output.pdf')
            output_doc.close()
        except Exception as e:
            print(e)
            pass

main()






