from ucm import UCMHMLCDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = UCMHMLCDataset(
    image_root="../../../data/raw/UCMerced_LandUse/Images",
    transform=transform,
)

print("Num label nodes : ", dataset.num_nodes)
print("Node names      : ", dataset.node_names)
print("Num samples     : ", len(dataset))
print("Parent map      : ", dataset.parent)
print("Depth map       : ", dataset.depth)

img, labels = dataset[0]
print("Image shape :", img.shape)
print("Label vector:", labels)
