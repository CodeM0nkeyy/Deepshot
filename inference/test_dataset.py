from inference.dataset import SoccerNetClipsTesting

dataset = SoccerNetClipsTesting("E:\\Projects\\hamza Saleem\\match4.mp4")

sample, size = dataset[0]

print("Returned size:", size)
print("Sample type:", type(sample))
