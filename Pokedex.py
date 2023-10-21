import pandas as pd
import numpy as np
import sqlite3
import json
import requests
import h5py

# wczytywanie obecnych danych zapisanych w pliku HDF5
def dataframe():
    with h5py.File('pokedex_history.hdf5', 'r') as file:
        # Wyświetlenie kluczy (grup) dostępnych w pliku
        # print("Dostępne grupy w pliku HDF5:")
        groups = list(file.keys())
        # print(groups)
        file.close()
    # hdf = pd.HDFStore('pokedex_history.hdf5', mode='r')
    # data = hdf.get('data')
    df = pd.read_hdf('pokedex_history.hdf5', keys='/data')
    # print(df.describe(datetime_is_numeric=True))
    name_list = list(df["name"])
    #print(name_list)
    return name_list    # zwraca listę nazwy spotkanych pokemonów

# pobranie danych o wybranym pokemonie z POKEAPI
def api(pokemon_name, df):
    url = 'https://pokeapi.co/api/v2/pokemon/'
    name = pokemon_name.lower()
    if name not in df["Name"]:
        req = requests.get(f"{url}{name}")  # działa
        if req.status_code == 200:          # gdy pobrano dobre dane z API
            pokemon = json.loads(req.text)
            hp = pokemon["stats"][0]["base_stat"]
            attack = pokemon["stats"][1]["base_stat"]
            defense = pokemon["stats"][2]["base_stat"]
            special_attack = pokemon["stats"][3]["base_stat"]
            special_defense = pokemon["stats"][4]["base_stat"]
            speed = pokemon["stats"][5]["base_stat"]
            type_1 = pokemon["types"][0]["type"]["name"]
            if len(pokemon["types"]) > 1:
                type_2 = pokemon["types"][1]["type"]["name"]
            else:
                type_2 = None
            new_data = np.array([name, hp, attack, defense, special_attack, special_defense, speed, type_1, type_2])
            df.loc[len(df)] = new_data      # dodanie nowych danych jako ostatni wiersz
            # print(df)
            return df   # zwraca zaktualizowany DataFrame

# funkcja sprawdzająca skuteczność ataku pokemona Attacker na pokemona Attacked
def attack_against(attacker: str, attacked: str, database: pd.DataFrame):
    # print(attacked)
    # type_1 = database.loc[database["Name"] == attacked, ["Type_1"]].values[0][0]
    # type_2 = database.loc[database["Name"] == attacked, ["Type_2"]].values[0][0]
    type_1_eff = 0
    rows_for_name = database.loc[database["Name"] == attacked]
    for index, row in rows_for_name.iterrows():     # pobranie wartości do obliczeń z DataFrame
        type_1 = row["Type_1"]
        type_2 = row["Type_2"]
        hp = row["HP"]
        attack = row["Attack"]
        defense = row["Defense"]
        special_attack = row["SP_Attack"]
        special_defense = row["SP_Defense"]
        speed = row["Speed"]
    # print(type_1, type_2)
    # type_att1 = database.loc[database["Name"] == attacker.lower(), ["Type_1"]].values[0][0]
    # type_att2 = database.loc[database["Name"] == attacker.lower(), ["Type_2"]].values[0][0]
    # print(type_att1, type_att2)
    conn = sqlite3.connect("pokemon_against.sqlite")  # połączenie do bazy danych - pliku
    if type_2 is not None:
        for row in conn.execute(f'SELECT against_{type_1}, against_{type_2} '\  
                                f'FROM against_stats WHERE against_stats.name = "{attacker}"'):
            type_1_eff = list(row)[0]   # wybranie wartości skuteczności na konkretny typ pokemona z bazy danych SQLite
            type_2_eff = list(row)[1]
            # print(type_1_eff, type_2_eff)
    else:
        for row in conn.execute(f'SELECT against_{type_1} '\
                                       f'FROM against_stats WHERE against_stats.name = "{attacker}"'):
            type_1_eff = list(row)[0]
            # print(type_1_eff)
    if type_1_eff != 0:     # sprawdzenie, czy pobrano z bazy dane o skuteczności
        effectiveness = type_1_eff * (int(attack) + int(hp) + int(special_attack) + int(speed) - int(defense) - \
                        int(special_defense)) + type_2_eff * (int(attack) + int(hp) + int(special_attack) + \
                        int(speed) - int(defense) - int(special_defense))
        print(f" Skutecznośc ataku {attacker} na {attacked} wynosi {effectiveness}")
        return effectiveness
    else:
        print("Brak danych")
        return None

if __name__ == "__main__":
    pokemon_list = dataframe()  # wczytanie listy nazw pokemonów
    df = pd.DataFrame(          # inicjalizacja DataFrame
        columns=["Name", "HP", "Attack", "Defense", "SP_Attack", "SP_Defense", "Speed", "Type_1", "Type_2"])
    for name in pokemon_list:   # pobranie danych o odwiedzonych pokemonach z API i dodanie do DataFrame
        api(name, df)
    print(df)

    for pokemon in pokemon_list:    # pętla sprawdzająca efektywnośc ataku dla wszystkich spotkanych pokemonów
        print(f"Twój pokemon {pokemon} walczy z: ")
        opponent = input('Wpisz nazwę z jakim chcesz walczyć: ')
        # opponent = "tentacool"
        api(opponent, df)
        attack_against(pokemon, opponent, df)


