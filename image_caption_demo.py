import sys
sys.path.append(r'./image_captioning/language_model/')
sys.path.append(r'./image_captioning/clip/')
from PIL import Image

import torch
from simctg import SimCTG
from clip import CLIP
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Language Model
# 加载语言模型
language_model_name = r'cambridgeltl/magic_mscoco'  # or r'/path/to/downloaded/cambridgeltl/magic_mscoco'
sos_token, pad_token = r'<-start_of_text->', r'<-pad->'
generation_model = SimCTG(language_model_name, sos_token, pad_token).to(device)
generation_model.eval()

# 加载 clip
model_name = r"openai/clip-vit-base-patch32"  # or r"/path/to/downloaded/openai/clip-vit-base-patch32"
clip = CLIP(model_name).to(device)
clip.eval()

sos_token = r'<-start_of_text->'
start_token = generation_model.tokenizer.tokenize(sos_token)
start_token_id = generation_model.tokenizer.convert_tokens_to_ids(start_token)
input_ids = torch.LongTensor(start_token_id).view(1, -1).to(device)

image_name_list2 = ['COCO_val2014_000000336777.jpg', 'COCO_val2014_000000182784.jpg', 'COCO_val2014_000000299319.jpg', 'COCO_val2014_000000516750.jpg',
                   'COCO_val2014_000000207151.jpg', 'COCO_val2014_000000078707.jpg', 'COCO_val2014_000000027440.jpg', 'COCO_val2014_000000033645.jpg',
                   'COCO_val2014_000000348905.jpg', 'COCO_val2014_000000545385.jpg', 'COCO_val2014_000000210032.jpg', 'COCO_val2014_000000577526.jpg']

# image_name_list2 = ['COCO_train2014_000000467840.jpg', 'COCO_train2014_000000533055.jpg',  'COCO_train2014_000000000531.jpg']

# image_name_list2 = ['test2.jpg']

image_name_list2 = ['COCO_val2014_000000581929.jpg', 'COCO_val2014_000000581913.jpg', 'COCO_val2014_000000581899.jpg', 'COCO_val2014_000000581887.jpg']

time_str = datetime.datetime.now().strftime('%Y_%m%d_%H_%M_%S')

# 将生成的caption写入到log文档
file = open("result/2_26_test2/{}log.txt".format(time_str), 'w')

k, alpha, beta, decoding_len = 45, 0.1, 2.0, 16
eos_token = '<|endoftext|>'
for image_name in image_name_list2:
    # image_path = r'./image_captioning/example_images/' + image_name
    image_path2 = r'../../dataset/coco/val2014/' + image_name
    # image_path2 = r'../../' + image_name
    image_instance = Image.open(image_path2)

    output = generation_model.magic_search(input_ids, k,
            alpha, decoding_len, beta, image_instance, clip, 60)
    print(output)
    file.write(output + '\n')
file.close()

# A street sign with a building in the background.
# A large cow standing in a street stall.
# A couple of people walking down a rainy street.
# A yellow boat is lined up on the beach.
# Large pizza with vegetables and cheese on a wooden table.
# A baseball player swinging a bat at a ball.
# A large giraffe standing in a zoo enclosure.
# A child playing with a disc in a backyard.
# A zooming person surfing on a wave in the ocean.
# A plate topped with cake and fork.
# A bird eating bread from a table.
# A cat laying on top of a bed.
