from ultralytics import YOLO

"""UNCOMMENT IF YOU WANT TO RETRAIN OR TRAIN THE MODEL FROM SCRATCH"""
if __name__ == '__main__':
    # load a pretrained model
    model = YOLO('yolo26n.pt') # latest YOLOG pretrained pytorch model

    # train the model
    results = model.train(
        data='../data/Multi-Crop Disease Dataset/Multicrop Disease Dataset/Multicrop Disease Dataset/data.yaml', 
        epochs=100,         # lower to <10 if want to run quick locally
        imgsz=640,
        device='cuda',      # ensure you have CUDA set up, otherwise use 'cpu'
        batch=8             # lower to 4 if not enough GPU memory
    )

    # export model
    model.export(
        format='engine'
    )

"""UNCOMMENT IF YOU RUN INTO AN ERROR IN THE LAST MAIN"""
# if __name__ == '__main__':
#     # resume model
#     model = YOLO("../runs/detect/train-6/weights/last.pt") # use the last training run to resume
#     results = model.train(resume=True)
#     # export model
#     model.export(
#         format='engine'
#     )

"""UNCOMMENT IF ONLY WANT TO EXPORT"""
# Only used after the model completed training
# if __name__ == 'main':
#     model = YOLO("runs/detect/train-6/weights/best.pt")
#     model.export(format='engine')