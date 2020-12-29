from matplotlib import pyplot as plt

with open("./output.txt", "r") as file:
    data = file.readlines()
    name_list = []
    time_list = []
    for text in data:
        if text == '\n':
            continue
        text = text.strip('\n')
        if text[3] == ':' or text[2] == ':' or text[6] == ':':
            name_list.append(text[:-2])
        if text[0:4] == "time":
            time_list.append(float(text[6:16]))
    plt.figure(dpi=128, figsize=(10, 6))
    plt.bar(name_list, time_list)
    plt.xlabel("Search Algorithm")
    plt.ylabel("Time")
    plt.show()
