import numpy as np
import matplotlib.pyplot as plt
def draw1(): # 柱状图
    to_draw = [(0.9459885358599089, 0.0013788484645537455, 0.052632615675537314), (0.9314009661835749, 0.0, 0.06859903381642507), (0.9764397905759162, 0.0, 0.023560209424083767), (0.5818965517241379, 0.02413793103448276, 0.39396551724137935), (0.8194335169158143, 0.003147128245476003, 0.17741935483870966), (0.6593516209476309, 0.0014962593516209476, 0.33915211970074816), (0.9560933818804883, 0.005140286999357464, 0.03876633112015422), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5126921701823383, 0.007865570253843403, 0.47944225956381836)]
    label10 = ['jeff.dasovich@enron.com', 'tana.jones@enron.com', 'steven.kean@enron.com', 'matthew.lenhart@enron.com', 'sara.shackleton@enron.com', 'kay.mann@enron.com', 'kate.symes@enron.com', 'pete.davis@enron.com', 'enron.announcements@enron.com', 'vince.kaminski@enron.com']
    for i in range(len(label10)):
        label10[i] = label10[i].split("@")[0].split(".")[0]
    label3 = ['enron','personal','other company']
    to_draw = np.array(to_draw).T
    prev = np.zeros(10)
    for i in range(3):
        plt.bar(range(10),to_draw[i], bottom = prev,label = label3[i],tick_label = label10)
        prev += to_draw[i]
    plt.legend()
    plt.savefig("email_bar_plot.png")
def draw2():
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="white")

    # Generate a large random dataset
    rs = np.random.RandomState(33)
    d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                    columns=list(ascii_letters[26:]))

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, annot=True,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig('z.png')
draw2()