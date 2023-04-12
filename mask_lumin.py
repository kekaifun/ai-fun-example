import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# change to your model path
sam_checkpoint = "tmp/sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
if torch.cuda.is_available():
    sam.to(device="cuda")
predictor = SamPredictor(sam_model=sam)

video = cv2.VideoCapture('video/lumin.mp4')
success, image = video.read()
height,width,layers = image.shape
size = (width,height)

input_point = np.array([[310, 150]])
input_label = np.array([1])
input_box = np.array([210, 40, 410, 350])


#plt.figure(figsize=(10,10))

fps = int(video.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('video/lumin_mask.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
count = 0
while success:
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # process image
    predictor.set_image(image)
    masks, scroes, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=True,
    )
    min_score = np.argmin(scroes)
    mask = masks[min_score]
    masked = image.copy()
    masked[mask != 0] = 0
    print(count)
    count += 1
    out.write(masked)
    if count == 10 :
        break
    success, image = video.read()

out.release()
video.release()
cv2.destroyAllWindows()
