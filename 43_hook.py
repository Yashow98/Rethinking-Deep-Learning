# @Author  : YashowHoo
# @File    : 43_hook.py
# @Description :
import torch
import timm

# print(timm.list_models("vgg*"))
model = timm.create_model(model_name="vgg11", pretrained=True)
input = torch.randn((1, 3, 224, 224))

print(model)

module_name = []
feature_in_hook = []
feature_out_hook = []

def hook(module, feature_input, feature_output):
    """
    Register forward hook

    :param module:
    :param feature_input:
    :param feature_output:
    :return:
    """
    print("Hooker working on!")

    module_name.append(module.__class__)
    feature_in_hook.append(feature_input[0].shape)  # feature_input is a tuple
    feature_out_hook.append(feature_output.shape)   # feature_output is a tensor


for layer in model.children():
    layer.register_forward_hook(hook=hook)

output = model(input)
print(module_name)
print(feature_in_hook)
print(feature_out_hook)

