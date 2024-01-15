import json
import streamlit as st
import torch
from pathlib import Path
from torchvision import transforms
from mega import Mega
import os

PATH = Path(__file__).parent


### HELPER FUNCTIONS ###
def get_data_specs(data_name):
    datas = {"agedb": {"age_num": 100}, "afad": {"age_num": 57}}
    age_num = datas[data_name]["age_num"]
    labels_txt = json.load(open("age_labels.txt"))
    labels = torch.tensor(labels_txt[data_name])
    return age_num, labels


def download_chpts(data):
    chpt_path = PATH / f"checkpoints/{data}.pt"

    if chpt_path.exists():
        pass

    else:
        chpt_ids = {"agedb": "https://mega.nz/file/hy9ngbhQ#dNsSTRuwDqov9Ykm4Dtsy_Gsgt76pz9D-FI_Tpw9qEA",
                    "afad": "https://mega.nz/file/h2dXHQBI#1vi3t5KnBussyXL9GN5pG3HGRtuLc5NUwDUYFe53f4E"}

        link = chpt_ids[data]
        mega = Mega()
        m = mega.login()
        warning = None
        try:
            warning = st.warning(f"Downloading checkpoints...")
            m.download_url(link)
            os.rename(f"{data}.pt", chpt_path)
        finally:
            if warning is not None:
                warning.empty()

    return chpt_path


def predict(image, model, age_labels, data):
    transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    
    img = image
    batch_t = torch.unsqueeze(transform(img), 0)
    model.eval()
    
    pred_age, pred_gen = model(batch_t)
    _, pred_gen_lbl = torch.max(pred_gen, 1)
    pred_gen_lbl = pred_gen_lbl.item()
    
    if pred_gen_lbl == 0:
        pred_gen = "Male"
    else:
        pred_gen = "Female"
        
    pred_age = torch.sum(pred_age * age_labels, dim=1)
    return pred_age, pred_gen, pred_gen_lbl
