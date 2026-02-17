import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

data = pd.read_csv('data/data1.csv', header=None)

data.columns = ['Długość działki kielicha', 'Szerokość działki kielicha', 'Długość płatka', 'Szerokość płatka', 'Gatunek']

setosa = data[data['Gatunek'] == 0]
versicolor = data[data['Gatunek'] == 1]
virginica = data[data['Gatunek'] == 2]
    
setosaCount = len(setosa)
versicolorCount = len(versicolor)
virginicaCount = len(virginica)
speciesCount = setosaCount + versicolorCount + virginicaCount

setosaPercentage = (setosaCount / speciesCount) * 100
versicolorPercentage = (versicolorCount / speciesCount) * 100
virginicaPercentage = (virginicaCount / speciesCount) * 100
speciesPercentage = ( speciesCount / speciesCount) * 100

print( "\n Tabela 1. Liczności gatunków irysów \n")
print("Gatunek      Liczebność(%)")
print(f"Setosa       {setosaCount} ({setosaPercentage:.1f}%)")
print(f"Versicolor   {versicolorCount} ({versicolorPercentage:.1f}%)")
print(f"Virginica    {virginicaCount} ({virginicaPercentage:.1f}%)")
print(f"Razem       {speciesCount} ({speciesPercentage:.1f}%)\n")

features = ['Długość działki kielicha', 'Szerokość działki kielicha', 'Długość płatka', 'Szerokość płatka']

minValues = []
maxValues = []
meanValues = []
medianValues = []
q1Values = []
q3Values = []
stdValues = []

for i, feature in enumerate(features):
    minValue = data[feature].min()
    maxValue = data[feature].max()
    meanValue = data[feature].mean()
    medianValue = data[feature].median()
    q1Value = data[feature].quantile(0.25)
    q3Value = data[feature].quantile(0.75)
    stdValue = data[feature].std(ddof=1)

    minValues.append(minValue)
    maxValues.append(maxValue)
    meanValues.append(meanValue)
    medianValues.append(medianValue)
    q1Values.append(q1Value)
    q3Values.append(q3Value)
    stdValues.append(stdValue)
    

print("Tabela 2. Charakterystyka cech irysów \n")
print("         Cecha                   Minimum     Śr. arytm. (± odch. stand.)    Mediana (Q1 - Q3)       Maksimum")
print(f"Długość działki kielicha (cm)     {minValues[0]:.2f}              {meanValues[0]:.2f} (±{stdValues[0]:.2f})            {medianValues[0]:.2f} ({q1Values[0]:.2f} - {q3Values[0]:.2f})        {maxValues[0]:.2f}")
print(f"Szerokość działki kielicha (cm)   {minValues[1]:.2f}              {meanValues[1]:.2f} (±{stdValues[1]:.2f})            {medianValues[1]:.2f} ({q1Values[1]:.2f} - {q3Values[1]:.2f})        {maxValues[1]:.2f}")
print(f"Długość płatka (cm)               {minValues[2]:.2f}              {meanValues[2]:.2f} (±{stdValues[2]:.2f})            {medianValues[2]:.2f} ({q1Values[2]:.2f} - {q3Values[2]:.2f})        {maxValues[2]:.2f}")
print(f"Szerokość płatka (cm)             {minValues[3]:.2f}              {meanValues[3]:.2f} (±{stdValues[3]:.2f})            {medianValues[3]:.2f} ({q1Values[3]:.2f} - {q3Values[3]:.2f})        {maxValues[3]:.2f} \n")

fig, axes = plt.subplots(4, 2, figsize=(8,9))

for i, feature in enumerate(features):
    bins = np.arange(math.floor(data[feature].min()), math.ceil(data[feature].max()) + 0.5, 0.5)
    if feature == "Szerokość działki kielicha" or feature == "Szerokość płatka":
        bins = np.arange(math.floor(data[feature].min()), math.ceil(data[feature].max())-0.25, 0.25)
    ax_hist = axes[i,0]
    counts = ax_hist.hist(data[feature], edgecolor="black", bins= bins)
    ax_hist.set_title(feature)
    if "Długość" in feature:
        ax_hist.set_xlabel("Długość")
    elif "Szerokość" in feature:
        ax_hist.set_xlabel("Szerokość")
    
    ax_hist.set_ylabel("Liczebność")
    amount = counts[0]
    max_height = max(amount)
    if feature == 'Szerokość działki kielicha' or feature == 'Szerokość płatka':
        ax_hist.set_yticks(np.arange(0, max_height + 10, 10))
    else:
        ax_hist.set_yticks(np.arange(0, max_height + 5, 5))
    if feature == 'Szerokość działki kielicha':
        ax_hist.set_xticks(np.arange(math.floor(data[feature].min()), math.ceil(data[feature].max()), 0.5))
    elif feature == 'Długość płatka':
        ax_hist.set_xticks(np.arange(math.floor(data[feature].min()), math.ceil(data[feature].max()) + 0.5, 1))
    elif feature == 'Szerokość płatka':
        ax_hist.set_xticks(np.arange(math.floor(data[feature].min()), math.ceil(data[feature].max()), 0.5))
    else:
        ax_hist.set_xticks(np.arange(math.floor(data[feature].min()), math.ceil(data[feature].max()) + 0.5, 0.5))




for i, feature in enumerate(features):
    ax_box = axes[i,1]

    setosa_data = setosa[feature]
    versicolor_data = versicolor[feature]
    virginica_data = virginica[feature]
    
    ax_box.boxplot([setosa_data, versicolor_data, virginica_data], labels=['setosa', 'versicolor', 'virginica'])
    ax_box.set_ylim(math.floor(float(data[feature].min())), math.ceil(float(data[feature].max())))
    ax_box.set_xlabel('Gatunek')
    if "Długość" in feature:
        ax_box.set_ylabel("Długość (cm)")
    elif "Szerokość" in feature:
        ax_box.set_ylabel("Szerokość (cm)")

plt.tight_layout()



def chart2 (data, featureX, featureY, ax = None):
    x = data[featureX]
    y = data[featureY]
    pearsonCorr = data.corr(method="pearson")
    correlation = pearsonCorr.loc[featureX, featureY]
    slope, intercept = np.polyfit(x,y,1)

    ax.scatter(x,y) 
    ax.plot(x, slope * x + intercept, color = 'red')
    ax.set_title(f"r = {correlation:.2f}; y = {slope:.1f}x + {intercept:.1f}")
    ax.set_xlabel(f"{featureX} (cm)")
    ax.set_ylabel(f"{featureY} (cm)")
    ax.set_xticks(np.arange(math.floor(x.min()), math.ceil(x.max())+1,1))


fig, axes = plt.subplots(3, 2, figsize=(8, 9))
chart2 (data, 'Długość działki kielicha', 'Szerokość działki kielicha',axes[0,0])
chart2 (data, 'Długość działki kielicha', 'Szerokość płatka',axes[1,0])
chart2 (data, 'Szerokość działki kielicha', 'Szerokość płatka',axes[2,0])
chart2 (data, 'Długość działki kielicha', 'Długość płatka',axes[0,1])
chart2 (data, 'Szerokość działki kielicha', 'Długość płatka',axes[1,1])
chart2 (data, 'Długość płatka', 'Szerokość płatka',axes[2,1]) 

plt.tight_layout()
plt.show()