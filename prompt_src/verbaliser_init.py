import pandas as pd


personalities = ['A','C','E','O','N']
for personality in personalities:
    df_verbalizer = pd.read_csv('big_five_cleaned.tsv', sep='\t')
    pos = [a.lower() for a in list(df_verbalizer['word'][df_verbalizer[personality]>0])]
    neg = [a.lower() for a in list(df_verbalizer['word'][df_verbalizer[personality]<0])]
    with open('big_five_' + personality + '.txt', 'w') as f:
        f.write(','.join(pos))
        f.write('\n')
        f.write(','.join(neg))
