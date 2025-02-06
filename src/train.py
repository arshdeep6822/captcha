import os
import torch
import glob
import numpy as np

from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection

import config
import dataset
import engine

from model import CaptchaModel
import torch.multiprocessing as mp

def run_training():
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
    print(image_files[:10])
    targets_orig = [x.split("/")[-1][:4] for x in image_files]
    # abcde -> [a b c d e]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    le = preprocessing.LabelEncoder()
    le.fit(targets_flat)
    targets_enc = [le.transform(x) for x in targets]
    targets_enc = np.array(targets_enc)+1
    print(targets_enc)
    print(le.classes_)
    
    train_imgs, test_imgs, train_targets, test_targets, train_orig_targets, test_orig_targets = model_selection.train_test_split(image_files, targets_enc, targets_orig, test_size=0.1, random_state=42)
    
    train_dataset = dataset.Classification(
        image_paths=train_imgs, 
        targets=train_targets, 
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    test_dataset = dataset.Classification(
        image_paths=test_imgs, 
        targets = test_targets, 
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    model = CaptchaModel(len(le.classes_))
    model.to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor = 0.8,
        verbose = True
    )
    
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(model, train_loader, optimizer)
        valid_preds, valid_loss = engine.eval_fn(model, train_loader)
        

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_training()