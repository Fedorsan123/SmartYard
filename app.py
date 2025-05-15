# app_resume.py

from ultralytics import YOLO
import os
from multiprocessing import freeze_support

def train_from_split(start_split=8, total_splits=20):
    splits_dir = 'E:/vehicle-splits'
    model_path = os.path.join('runs', 'split_7', 'weights', 'last.pt')

    epochs_all = [20] + [18] * (total_splits - 1)

    for i in range(start_split, total_splits + 1):
        split     = f"split_{i}"
        split_dir = os.path.join(splits_dir, split)
        data_yaml = os.path.join(split_dir, 'dataset.yaml')
        epochs    = epochs_all[i - 1]

        # Если нет dataset.yaml — создаём
        if not os.path.isfile(data_yaml):
            with open(data_yaml, 'w') as f:
                f.write(
                    f"path: {split_dir}\n"
                    "train: images\n"
                    "val:   images\n"
                    "nc: 1\n"
                    "names: ['vehicle']\n"
                )
            print(f"📝 Created {data_yaml}")

        print(f"\n▶️  Training {split} for {epochs} epochs — loading weights from {model_path}")

        # Тренировка
        model = YOLO(model_path)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=4,
            optimizer='AdamW',
            lr0=0.001,        # пониженный стартовый LR для плавности
            lrf=0.01,
            single_cls=True,
            workers=4,
            augment=False,
            save_period=5,
            project='runs',
            name=split,
            resume=False,      # подхватываем состояние оптимизатора из last.pt
            amp=False,
            cache=False
        )

        # Обновляем путь к весам для следующего сплита
        best = os.path.join('runs', split, 'weights', 'best.pt')
        last = os.path.join('runs', split, 'weights', 'last.pt')
        if os.path.isfile(last):   # last.pt содержит optimizer-state
            model_path = last
        elif os.path.isfile(best):
            model_path = best
        else:
            raise FileNotFoundError(f"No weights found in runs/{split}/weights")
        print(f"✅ Finished {split}, next start from {model_path}")

if __name__ == '__main__':
    freeze_support()
    train_from_split(start_split=8, total_splits=20)
