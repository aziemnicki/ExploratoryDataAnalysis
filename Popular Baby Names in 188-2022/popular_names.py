import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3


def zad2():
    unique = df["Name"].unique()
    print("Liczba unikalnych imion: ", len(unique))  # zadanie 2 unikalne imiona
    unique_men = df.loc[df["Sex"] == 'M', :]
    men_len = len(unique_men["Name"].unique())
    print("Liczba unikalnych imion męskich: ", men_len)  # zadanie 3 unikalne imiona męskie i żeńskie
    unique_women = df.loc[df["Sex"] == 'F', :]
    women_len = len(unique_women["Name"].unique())
    print("Liczba unikalnych imion żeńskich: ", women_len)


def zad4():
    df_men = df[df['Sex'] == 'M']
    df_women = df[df['Sex'] == 'F']
    df['frequency_male'] = df_men['Frequency'] / df_men.groupby('Year')['Frequency'].transform('sum')
    df['frequency_female'] = df_women['Frequency'] / df_women.groupby('Year')['Frequency'].transform('sum')
    df['frequency_male'] = df['frequency_male'].fillna(0)  # usuwanie NaN
    df['frequency_female'] = df['frequency_female'].fillna(0)


def zad5():
    summ = df.pivot_table(index='Year', columns='Sex', values='Frequency', aggfunc='sum')
    born = summ['M'] + summ['F']
    ratio = summ['F'] / summ['M']
    # wykres
    width = 0.5
    x = np.arange(1880, 2023, 1)
    fig, axs = plt.subplots(2, 1)
    axs[0].bar(x, born, width, label='Liczba urodzeń')
    axs[0].set_xticks(x)
    axs[0].set_yticks(np.arange(0, max(born), 300000))
    axs[0].set_xticklabels(years)
    axs[0].tick_params(axis='x', labelrotation=90)
    axs[0].legend()

    axs[1].bar(x, ratio, width, label='Stosunek liczby dziewczynek do chłopców')
    axs[1].set_xticks(x)
    axs[1].set_yticks(np.arange(0, max(ratio) + 0.1, 0.1))
    axs[1].set_xticklabels(years)
    axs[1].tick_params(axis='x', labelrotation=90)
    axs[1].legend()
    plt.show()
    max_ratio = max(ratio)
    min_ratio = min(ratio, key=lambda z: abs(z - 1))
    for i, year in enumerate(ratio):  # odpowiedz na pytanie 5
        if year == min_ratio:
            print(f'Najmniejsza różnica w liczbie urodzin dziewczynek do chłopców zanotowano w {years[i]} i\
             wynosi {min_ratio}')
        elif year == max_ratio:
            print(f'Największą różnicę w liczbie urodzin dziewczynek do chłopców zanotowano w {years[i]} i\
             wynosi {max_ratio}')


def zad6():
    top_1000 = df.groupby(['Name', 'Sex'], as_index=False)[['frequency_male', 'frequency_female', 'Frequency']].sum()
    top_1000.sort_values(['frequency_male', 'frequency_female', ], axis=0, inplace=True, ascending=False)
    top_1000_male = top_1000.loc[top_1000['Sex'] == 'M', :].head(1000)
    top_1000_female = top_1000.loc[top_1000['Sex'] == 'F', :].head(1000)
    # print(top_1000_male.head(1000))
    # print(top_1000_female.head(1000))
    return top_1000_male, top_1000_female


def zad7(man, woman):
    first_men = man.iloc[0]['Name']
    john = df.loc[((df['Name'] == first_men) & (df['Sex'] == 'M')), :]
    first_woman = woman.iloc[0]['Name']
    woman_name = df.loc[(df['Name'] == first_woman) & (df['Sex'] == 'F'), :]
    print(
        f'Imię John nadano w latach {years[54]}, {years[100]}, {years[-1]}, kolejno {john.iloc[53]["Frequency"]}, \
        {john.iloc[99]["Frequency"]}, {john.iloc[142]["Frequency"]} razy.')
    # print(john)
    x = np.arange(1880, 2023, 1)
    fig, axs = plt.subplots(2, 1)
    label1 = axs[0].plot(x, john['Frequency'], label='Najpopularniejszy mężczyzna = John')
    axs[0].set_xticks(x)
    axs[0].set_ylabel('Liczba nadań imienia John')
    axs[0].set_yticks(np.arange(0, max(john['Frequency']), 5000))
    axs[0].set_xticklabels(years)
    axs[0].tick_params(axis='x', labelrotation=90)
    # axs[0].legend()
    ax2 = axs[0].twinx()
    label2 = ax2.plot(x, john['frequency_male'], color='orange', label='Wartość popularności imienia John')
    ax2.set(ylabel='Popularność (frequency_male)')
    ax2.set_yticks(np.arange(0, max(john['frequency_male']), 0.01))
    labels = label1 + label2
    labs = [l.get_label() for l in labels]
    ax2.legend(labels, labs, loc='upper right')
    x = np.arange(1880, 2023, 1)

    label3 = axs[1].plot(x, woman_name['Frequency'], label='Najpopularniejsza kobieta')
    axs[1].set_xticks(x)
    axs[1].set_yticks(np.arange(0, max(john['Frequency']), 10000))
    axs[1].set_ylabel('Liczba nadań imienia Mary')
    axs[1].set_xticklabels(years)
    axs[1].tick_params(axis='x', labelrotation=90)
    # axs[1].legend()

    ax4 = axs[1].twinx()
    label4 = ax4.plot(x, woman_name['frequency_female'], color='orange', label='Wartość popularności imienia Mary')
    ax4.set_ylabel('Popularność (frequency_male)')
    ax4.set_yticks(np.arange(0, max(woman_name['frequency_female']), 0.01))
    ax4.legend(loc='upper right')
    labels = label3 + label4
    lab = [l.get_label() for l in labels]
    ax4.legend(labels, lab)

    plt.show()


def zad8(men, woman):
    summ = df.pivot_table(index='Year', columns='Sex', values='Frequency', aggfunc='sum')
    # print(suma)
    names_man = df.loc[((df['Name'].isin(men['Name'])) & (df['Sex'] == 'M')), :]
    names_woman = df.loc[((df['Name'].isin(woman['Name'])) & (df['Sex'] == 'F')), :]
    names_top = pd.concat([names_man, names_woman])
    top_1000 = names_top.pivot_table(index='Year', columns=['Sex'], values=['Frequency'], aggfunc='sum')
    top_1000_men_percent = top_1000[('Frequency', 'M')] / summ['M'] * 100
    top_1000_female_percent = top_1000[('Frequency', 'F')] / summ['F'] * 100
    diff = top_1000_men_percent - top_1000_female_percent
    sort = diff.sort_values(ascending=False)
    print(f'Największa różnorodnośc wśród imion jest w roku {sort.index[0]}, i wynosi {sort.iloc[0]} ')
    x = np.arange(1880, 2023, 1)
    fig, axs = plt.subplots(1, 1)
    label1 = axs.plot(x, top_1000_men_percent, label='Procent mężczyzn z top1000')
    label2 = axs.plot(x, top_1000_female_percent, label='Procent kobiet z top1000')
    labels = label1 + label2
    lab = [l.get_label() for l in labels]
    axs.set_xticks(x)
    axs.set_yticks(np.arange(50, 100, 1))
    axs.set_ylim(50, 100)
    axs.set_xticklabels(years)
    axs.tick_params(axis='x', labelrotation=90)
    axs.legend(labels, lab)

    plt.show()


def zad9():
    df['letter'] = df['Name'].str[-1]
    last_letter = df.groupby(['Year', 'Sex', df['letter']], as_index=False)['Frequency'].sum()
    year_group = last_letter.loc[
                 (last_letter['Year'] == '1917') | (last_letter['Year'] == '1967') | (last_letter['Year'] == '2022'), :]
    suma = year_group.groupby(['Year'], as_index=False)['Frequency'].sum()
    letter_men = year_group.loc[year_group['Sex'] == "M", :].copy()
    men_list = ['1917', '1967', '2022']
    letter_men.loc[78, :] = ['1917', 'M', 'q', 0]
    width = 0.25
    fig, axs = plt.subplots(1, 1)
    letters = letter_men['letter'].unique()
    for i, year in enumerate(men_list):
        summ = int(suma.iloc[i]['Frequency'])
        x = np.arange(0, len(letter_men['letter'].unique()), 1)
        norm = letter_men.loc[letter_men['Year'] == year, :]['Frequency'] / int(summ)
        axs.bar(x + width * i, norm, width, label=f'Rok {year}')
    axs.set_xticks(x + width)
    axs.set_xticklabels(letter_men['letter'].unique())
    axs.set_ylabel('Liczba nadań imion męskich')
    axs.tick_params(axis='x', labelrotation=0)
    axs.legend()

    change = dict()
    for i, letter in enumerate(letters):
        change[i] = max(letter_men.loc[letter_men['letter'] == letter, :]['Frequency']) - min(
            letter_men.loc[letter_men['letter'] == letter, :]['Frequency'])
    sorted_items = [(key, change[key]) for key in sorted(change, key=change.get, reverse=True)]
    # print(sorted_items)
    print(f"Największy wzrost/spadek między 1917 a 2022 zaobserwowano dla litery {letter_men.iloc[13]['letter']}")

    letter_n = last_letter.loc[((last_letter['letter'] == 'n') & (last_letter['Sex'] == 'M')), :]['Frequency']
    letter_d = last_letter.loc[((last_letter['letter'] == 'd') & (last_letter['Sex'] == 'M')), :]['Frequency']
    letter_z = last_letter.loc[((last_letter['letter'] == 'y') & (last_letter['Sex'] == 'M')), :]['Frequency']
    fig, axs = plt.subplots(1, 1)
    x = np.arange(1880, 2023, 1)
    axs.plot(x, letter_n, color='blue', label='n')
    axs.plot(x, letter_d, color='orange', label='d')
    axs.plot(x, letter_z, color='green', label='z')

    axs.set_xticks(x)
    axs.set_xticklabels(years)
    axs.set_ylabel('Liczba nadań imion męskich')
    axs.set_yticks(np.arange(0, max(last_letter['Frequency']), 30000))
    axs.tick_params(axis='x', labelrotation=90)
    axs.legend()

    plt.show()


def zad10(man, woman):
    both = pd.merge(man, woman, on='Name', how='inner')
    df['Year'] = df['Year'].astype(int)
    early = df.loc[df['Year'] <= 1930, :]
    later = df.loc[df['Year'] >= 2000, :]
    both_early = pd.merge(both, early, on="Name", how='inner')

    # do 1930
    both_early = both_early.groupby(['Name'])[['Frequency_x', 'Frequency_y']].sum()
    both_later = pd.merge(both, later, on="Name", how='inner')

    # po 2000
    both_later = both_later.groupby(['Name'])[['Frequency_x', 'Frequency_y']].sum()
    change_ratio_early = dict(zip(both_early.index, abs(both_early['Frequency_x'] - both_early['Frequency_y'])))
    sort_ratio = sorted(change_ratio_early.items(), key=lambda x: x[1], reverse=True)
    print(f"Imiona z największą zmianą imon męskich do żeńskich do 1930 roku to {sort_ratio[0]} i {sort_ratio[1]}")
    change_ratio_later = dict(zip(both_later.index, abs(both_later['Frequency_x'] - both_later['Frequency_y'])))
    sort_ratio = sorted(change_ratio_later.items(), key=lambda x: x[1], reverse=True)
    print(f"Imiona z największą zmianą imon męskich do żeńskich od 2000 roku to {sort_ratio[0]} i {sort_ratio[1]}")

    james = df.loc[(df['Name'] == 'James') & (df['Year'] <= 1930) & (df['Sex'] == 'M'), :]
    john = df.loc[(df['Name'] == 'John') & (df['Year'] <= 1930) & (df['Sex'] == 'M'), :]

    x = np.arange(1880, 1931, 1)
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x, james['Frequency'], label='Liczba urodzeń imienia James')
    axs[0].plot(x, john['Frequency'], label='Liczba urodzeń imienia John')
    axs[0].set_xticks(x)
    axs[0].set_yticks(np.arange(0, max(max(john['Frequency']), max(james['Frequency'])), 3000))
    axs[0].set_xticklabels(x)
    axs[0].tick_params(axis='x', labelrotation=90)
    axs[0].legend()

    james = df.loc[(df['Name'] == 'James') & (df['Year'] >= 2000) & (df['Sex'] == 'M'), :]
    john = df.loc[(df['Name'] == 'John') & (df['Year'] >= 2000) & (df['Sex'] == 'M'), :]
    x = np.arange(2000, 2023, 1)
    axs[1].plot(x, james['Frequency'], label='Liczba urodzeń imienia James')
    axs[1].plot(x, john['Frequency'], label='Liczba urodzeń imienia John')
    axs[1].set_xticks(x)
    axs[1].set_yticks(np.arange(0, max(max(john['Frequency']), max(james['Frequency'])), 3000))
    axs[1].set_xticklabels(x)
    axs[1].tick_params(axis='x', labelrotation=90)
    axs[1].legend()
    plt.show()


def zad11():
    conn = sqlite3.connect("data/demography_us_2023.sqlite3")

    query = """
    SELECT p.Year, p.Age, p.Female, p.Male, p.Total, d.Year AS Death_Year, d.Female AS Deaths_Female, \
    d.Male AS Deaths_Male, d.Total as Deaths_Total  
    FROM population p
    INNER JOIN deaths d ON 
        p._rowid_ = d._rowid_
    WHERE p.Year >= 1935 AND p.Year <= 2020 
    ORDER BY p.Year;
    """
    birth = """
    SELECT * FROM births
    WHERE Year >= 1935
    ORDER BY Year;
    """
    births = pd.read_sql_query(birth, conn)
    # print(type(births))
    pop = pd.read_sql_query(query, conn)
    population = pd.DataFrame(pop)
    conn.close()
    # print(population)

    # # zadanie 12
    przyrost = population.groupby(['Death_Year'], as_index=False).sum()
    increse = births['Total'] - przyrost['Deaths_Total']
    # print(przyrost)
    width = 0.5
    x = np.arange(1935, 2022, 1)
    fig, ax = plt.subplots()
    ax.bar(x, increse, width, label='Przyrost naturalny')
    ax.set_xticks(x)
    ax.set_yticks(np.arange(min(increse), max(increse), 100000))
    ax.set_xticklabels(x)
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend()
    plt.show()

    # zadanie 13
    rate = population.loc[(population['Age'] == '1'), :]
    # print(rate)
    year_list = list(pop['Death_Year'].unique())
    # print(year_list)
    survival_rate = []
    for i, year in enumerate(year_list):
        if 1 < i < len(year_list):
            survival_rate.append((list(births.loc[births['Year'] == (year - 1)]['Total'])[0] -
                                  list(rate.loc[rate['Death_Year'] == year]['Deaths_Total'])[0])
                                 / list(births.loc[births['Year'] == (year - 1)]['Total'])[0] * 100)
    # print(survival_rate)
    x = np.arange(1936, 2021, 1)
    fig, ax = plt.subplots()
    ax.plot(x, survival_rate, label='Współczynnik przeżywalności dzieci w 1 r.ż.')
    ax.set_xticks(x)
    ax.set_yticks(np.arange(99.0, 100, 0.05))
    ax.set_ylim(99.2, max(survival_rate) + 0.01)
    ax.set_xticklabels(x)
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend()
    plt.show()

    return births, population


def zad14(births, population):
    summ = df.pivot_table(index='Year', columns='Sex', values='Frequency', aggfunc='sum')
    period_sum = summ.loc[(summ.index >= 1935) & (summ.index < 2021), :]
    alls = period_sum['F'] + period_sum['M']
    alls = alls.reset_index()
    przyrost = population.groupby(['Death_Year'], as_index=False).sum()
    increse = births['Total'] - przyrost['Deaths_Total']
    total_births = increse.iloc[0:86]
    df_increse = alls[0] - przyrost.iloc[0:86]['Deaths_Total']
    diff = total_births - df_increse
    max_diff = max(diff)
    min_diff = min(diff)
    x = np.arange(1935, 2021, 1)
    error = abs(diff) / ((df_increse + total_births) / 2)
    fig, ax = plt.subplots()
    ax.plot(x, error, label='Liczba urodzeń z bazy danych')
    ax.set_xticks(x)

    ax.set_xticklabels(x)
    ax.tick_params(axis='x', labelrotation=90)
    ax.legend()

    for i, year in enumerate(diff):
        if year == min_diff:
            print(
                f'Najmniejsza różnica w liczbie urodzin dziewczynek do chłopców zanotowano w {years[i]} i \
                wynosi {min_diff}')
        elif year == max_diff:
            print(
                f'Największą różnicę w liczbie urodzin dziewczynek do chłopców zanotowano w {years[i]} i \
                wynosi {max_diff}')
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_list = []
    years = []
    files = os.listdir('data')
    files = sorted(files)
    for file in files:
        if file.endswith('.txt'):
            data = pd.read_csv(os.path.join('data', file), sep=',', header=None)
            data['Year'] = file[3:-4]
            file_list.append(data)
            years.append(file[3:-4])

    df = pd.concat(file_list, axis=0, ignore_index=True)
    rename = {0: 'Name', 1: 'Sex', 2: 'Frequency'}
    df.rename(columns=rename, inplace=True)
    # print(df)
    zad2()
    zad4()
    zad5()
    top_men, top_woman = zad6()
    zad7(top_men, top_woman)
    zad8(top_men, top_woman)
    zad9()
    zad10(top_men, top_woman)
    new_birth, pop = zad11()
    zad14(new_birth, pop)
