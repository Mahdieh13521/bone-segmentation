import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import Knee_dataset
from model import UNetLext
from args import get_args
import cv2

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv = os.path.join(args.csv_dir, "test.csv")
    df_test = pd.read_csv(test_csv)

    test_dataset = Knee_dataset(df_test, test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNetLext(
        input_channels=1,
        output_channels=1,
        pretrained=False,
        path_pretrained='',
        restore_weights=False,
        path_weights=''
    )

    state_dict = torch.load(args.path_weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    os.makedirs("predictions", exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img = batch['image'].to(device)
            out = torch.sigmoid(model(img))         
            out_np = (out.squeeze().cpu().numpy() * 255).astype('uint8')

            filename = os.path.basename(batch['xrays'][0] if 'xrays' in batch else f"img_{i}.png")
            cv2.imwrite(os.path.join("predictions", filename), out_np)

    print("Inference completed. Predicted masks saved in ./predictions")

if __name__ == "__main__":
    main()
