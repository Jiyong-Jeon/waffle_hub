if __name__ == "__main__":
        from ultralytics import YOLO
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        try:
            model = YOLO("yolov8n.pt", task="detect")
            model.train(
                **{'data': '/home/snuailab/Desktop/waffle_hub/docs/tutorials/datasets/mnist_det/exports/ULTRALYTICS/data.yaml', 'epochs': 50, 'batch': 4, 'imgsz': [640, 640], 'lr0': 0.01, 'lrf': 0.01, 'rect': False, 'device': '0', 'workers': 2, 'seed': 0, 'verbose': True, 'project': 'hubs/ultralytics_mnist_detection', 'name': 'artifacts'}
            )
        except Exception as e:
            print(e)
            raise e
        