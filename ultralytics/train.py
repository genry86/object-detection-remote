from ultralytics import YOLO, checks, hub
checks()

hub.login('99a418153043b80ca775c08345e817129d11153ee1')

model = YOLO('https://hub.ultralytics.com/models/8A3MTreSK5yZvvRESRQp')
results = model.train()