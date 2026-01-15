import pandas as pd
from tdc.single_pred import ADME

data = ADME(name="Lipophilicity_AstraZeneca")
split = data.get_split()
data.get_data()

df = pd.read_csv("data/lipophilicity_astrazeneca.tab", sep="\t")
