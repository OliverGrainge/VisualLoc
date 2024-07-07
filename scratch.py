from PlaceRec.Datasets import AmsterTime, SVOX, Pitts30k_Val, MapillarySLS

amstertime = AmsterTime()
import pickle

with open("dataset_sizes.pkl", "rb") as f:
    sizes = pickle.load(f)


print(amstertime.name, len(amstertime.map_paths))
sizes["AmsterTime"] = len(amstertime.map_paths)

ds = MapillarySLS()
print(ds.name, len(ds.map_paths))
sizes["MapillarySLS"] = len(ds.map_paths)

ds = SVOX()
print(ds.name, len(ds.map_paths))
sizes["SVOX"] = len(ds.map_paths)


ds = Pitts30k_Val()
print(ds.name, len(ds.map_paths))
sizes["Pitts30k_Val"] = len(ds.map_paths)

print(sizes)

with open("dataset_sizes.pkl", "wb") as f:
    pickle.dump(sizes, f)
