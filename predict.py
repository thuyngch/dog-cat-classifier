import torch, os
from utils.helpers import *
import warnings
from PIL import Image
from torchvision import transforms
from glob import glob

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@torch.no_grad()
def image_transform(imagepath):
	test_transforms = transforms.Compose([
		transforms.Resize(255),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	image = Image.open(imagepath)
	imagetensor = test_transforms(image)
	return imagetensor

@torch.no_grad()
def predict(model, imagepath, device='cpu'):
	image = image_transform(imagepath)
	image1 = image[None,:,:,:].to(device)
	ps = torch.exp(model(image1))
	topconf, topclass = ps.topk(1, dim=1)
	if topclass.item() == 1:
		return {'class':'dog','confidence':str(topconf.item())}
	else:
		return {'class':'cat','confidence':str(topconf.item())}

if __name__ == '__main__':
	# Define
	img_dir = "data"
	model_path = 'models/catvdog.pth'

	# Build model
	model = load_model(model_path)
	model.eval()
	# model.eval().cuda()

	# Infer images
	image_files = sorted(glob(os.path.join(img_dir, "*.*")))
	for image_file in image_files:
		result = predict(model, image_file)
		print("{}: {}".format(image_file, result))
