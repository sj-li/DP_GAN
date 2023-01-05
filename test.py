
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
import time
import torch


#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

#--- create models ---#
model = models.DP_GAN_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

total_time = 0
#--- iterate over validation set ---#
for i, data_i in enumerate(dataloader_val):
    _, label = models.preprocess_input(opt, data_i)
    end = time.time()
    generated = model(None, label, "generate", None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t = time.time() - end
    total_time += t
    image_saver(label, generated, data_i["name"])

print("Avg time: ", total_time/len(dataloader_val))
