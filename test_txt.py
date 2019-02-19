import json

labels_file = 'labels.json'

labels = {0: '1x4LBlack', 1: '1x4LRed', 2: '3x5LBlack', 3: '3x5LGray', 4: '3x5LGreen', 5: '3x5LOrange', 6: '3x5LRed', 7: 'Gear20Beige', 8: 'Pin3Blue', 9: 'PinBlack'}




with open(labels_file, 'w') as fp:
    json.dump(labels, fp)

with open(labels_file) as f:
    data = json.load(f,object_hook=jsonKeys2int)

