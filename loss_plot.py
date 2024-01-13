# ploting the loss from the training log file
import matplotlib.pyplot as plt
name = 'model_TEST' #input("Enter the name of the log file (like model_TEST): ")

with open(f'/Volumes/AI-Models/AI-Models/my-trained-models/{name}.log', 'r') as f:
    # data writed as separate lines with numbers of loss on each line, calculate lines and draw plot
    data = f.readlines()
    x = []
    y = []
    for i in range(len(data)):
        x.append(i)
        y.append(float(data[i]))
    plt.plot(x, y)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f'Loss plot for {name}')
    plt.get_current_fig_manager().set_window_title(f"Kovalev's GPT stat.")
    plt.show()
