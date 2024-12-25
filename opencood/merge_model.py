import torch

state_dict_model1 = torch.load('')
state_dict_model2 = torch.load('')

for key in state_dict_model1:
    if key in state_dict_model2:
        print(key)
        state_dict_model1[key] = (state_dict_model1[key] + state_dict_model2[key]) / 2


for key in state_dict_model2:
    if key not in state_dict_model1:
        state_dict_model1[key] = state_dict_model2[key]
        
torch.save(state_dict_model1, '')
