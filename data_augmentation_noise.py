import random
import pandas as pd
import numpy as np

archivo = 'C:/Users/Miguel/Desktop/TFG/Code/LaLiga.xlsx'

df = pd.read_excel(archivo)

for i in range(3143):

    # seleccionamos la fila aleatoria
    # python quita la cabecera (fila 1 en excel) y cuenta la primera fila como la 0 (fila 2 en excel) --> 1049 - 2 = 1047
    row_player = random.randint(0, 1047)
    row_goalkeeper = random.randint(0, 1047)


    # Nombres jugadores y equipos
    name_player = df.values[row_player, 3]
    name_goalkeeper = df.values[row_goalkeeper, 4]


    # jornada
    match_week = random.randint(1, 38)


    # minuto del penalti
    time_penalty = random.randint(1, 100)


    # resultado
    goles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = [0.305, 0.3, 0.2, 0.1, 0.025, 0.02, 0.015, 0.015, 0.015, 0.005]
    team_local = random.choices(goles, weights)
    team_visitor = random.choices(goles, weights)

    team_local = int(''.join(map(str, team_local)))
    team_visitor = int(''.join(map(str, team_visitor)))


    # Para saber qué equipo es local o visitante
    home_away = random.randint(0, 1)

    if home_away == 0:
        local = df.values[row_player, 5]
        visitor = df.values[row_goalkeeper, 6]
    else:
        local = df.values[row_goalkeeper, 6]
        visitor = df.values[row_player, 5]


    # dirección del penalti
    direction = ['LT', 'CT', 'RT', 'LC', 'CC', 'RC', 'LD', 'CD', 'RD']


    # número de veces a cada dirección
    player_direction = [[0,0,0],
                        [0,0,0],
                        [0,0,0]]

    gk_direction = [[0,0,0],
                    [0,0,0],
                    [0,0,0]]


    # pie con el que dispara
    foot = df.values[row_player, 12]


    # fecha
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2012, 2021)

    if month >= 8:
        season = ('%d/%d' % (year, year + 1))
    else:
        season = ('%d/%d' % (year - 1, year))


    # gol / parada / fuera
    player_goal = 0
    player_total_penal = 0
    player_probability_goal = []

    gk_saved = 0
    gk_total_penal = 0
    gk_probability_saved = []

    for i in df.index:

        if name_player == df.values[i, 3]:

            if df.values[i, 13] == 'LT':
                player_direction[0][0] += 1
            elif df.values[i, 13] == 'CT':
                player_direction[0][1] += 1
            elif df.values[i, 13] == 'RT':
                player_direction[0][2] += 1
            elif df.values[i, 13] == 'LC':
                player_direction[1][0] += 1
            elif df.values[i, 13] == 'CC':
                player_direction[1][1] += 1
            elif df.values[i, 13] == 'RC':
                player_direction[1][2] += 1
            elif df.values[i, 13] == 'LD':
                player_direction[2][0] += 1
            elif df.values[i, 13] == 'CD':
                player_direction[2][1] += 1
            elif df.values[i, 13] == 'RD':
                player_direction[2][2] += 1

            player_total_penal += 1

            if df.values[i, 10] == 1:
                player_goal += 1
                player_probability_goal.append(1)
            else:
                player_probability_goal.append(0)

        if name_goalkeeper == df.values[i, 4]:
            if df.values[i, 13] == 'LT':
                gk_direction[0][0] += 1
            elif df.values[i, 13] == 'CT':
                gk_direction[0][1] += 1
            elif df.values[i, 13] == 'RT':
                gk_direction[0][2] += 1
            elif df.values[i, 13] == 'LC':
                gk_direction[1][0] += 1
            elif df.values[i, 13] == 'CC':
                gk_direction[1][1] += 1
            elif df.values[i, 13] == 'RC':
                gk_direction[1][2] += 1
            elif df.values[i, 13] == 'LD':
                gk_direction[2][0] += 1
            elif df.values[i, 13] == 'CD':
                gk_direction[2][1] += 1
            elif df.values[i, 13] == 'RD':
                gk_direction[2][2] += 1

            gk_total_penal += 1

            if df.values[i, 11] == 1:
                gk_saved += 1
                gk_probability_saved.append(1)
            else:
                gk_probability_saved.append(0)


    # ruido gaussiano jugador
    player_direction = np.array(player_direction, float)

    media = player_direction.mean()
    desv_tipica = player_direction.std()

    noise = np.random.normal(media, desv_tipica, [3, 3])

    player_direction = player_direction + noise

    for columna in range(len(player_direction[0])):
        for fila in range(len(player_direction)):
            if player_direction[fila][columna] < 0.0:
                player_direction[fila][columna] = abs(player_direction[fila][columna])
                # player_direction[fila][columna] = 0.0


    # ruido gaussiano portero
    keeper_direction = np.array(gk_direction, float)

    media = keeper_direction.mean()
    desv_tipica = keeper_direction.std()

    noise = np.random.normal(media, desv_tipica, [3, 3])

    keeper_direction = keeper_direction + noise

    for columna in range(len(keeper_direction[0])):
        for fila in range(len(keeper_direction)):
            if keeper_direction[fila][columna] < 0.0:
                keeper_direction[fila][columna] = abs(keeper_direction[fila][columna])
                # player_direction[fila][columna] = 0.0


    # Probabilidad kick_direction
    probabilities_player = 0
    probabilities_direction = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for columna in range(len(player_direction[0])):
        for fila in range(len(player_direction)):
            probabilities_player += player_direction[fila][columna]

    i = 0
    suma = 0.0
    for columna in range(len(player_direction[0])):
        for fila in range(len(player_direction)):
            probabilities_direction[i] = player_direction[fila][columna]/probabilities_player
            suma = suma + probabilities_direction[i]
            i += 1

    if suma > 1:
        max_item = max(probabilities_direction, key=float)
        for j in range(len(probabilities_direction)):
            if probabilities_direction[j] == max_item:
                probabilities_direction[j] = probabilities_direction[j] - (suma - 1)
            j += 1
    elif suma < 1:
        max_item = max(probabilities_direction, key=float)
        for j in range(len(probabilities_direction)):
            if probabilities_direction[j] == max_item:
                probabilities_direction[j] = probabilities_direction[j] + (1 - suma)
            j += 1

    kick_direction = np.random.choice(direction, p=probabilities_direction)


    # Probabilidad keeper_direction
    probabilities_gk = 0
    probabilities_direction = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for columna in range(len(keeper_direction[0])):
        for fila in range(len(keeper_direction)):
            probabilities_gk += keeper_direction[fila][columna]

    i = 0
    suma = 0.0
    for columna in range(len(keeper_direction[0])):
        for fila in range(len(keeper_direction)):
            probabilities_direction[i] = keeper_direction[fila][columna]/probabilities_gk
            suma = suma + probabilities_direction[i]
            i += 1

    if suma > 1:
        max_item = max(probabilities_direction, key=float)
        for j in range(len(probabilities_direction)):
            if probabilities_direction[j] == max_item:
                probabilities_direction[j] = probabilities_direction[j] - (suma - 1)
            j += 1
    elif suma < 1:
        max_item = max(probabilities_direction, key=float)
        for j in range(len(probabilities_direction)):
            if probabilities_direction[j] == max_item:
                probabilities_direction[j] = probabilities_direction[j] + (1 - suma)
            j += 1

    gk_direction = np.random.choice(direction, p=probabilities_direction)


    # Ruido gaussiano gol / parada / fuera

    # - Probabilidad gol
    player_probability_goal = np.array(player_probability_goal, float)

    media = player_probability_goal.mean()
    desv_tipica = player_probability_goal.std()

    noise = np.random.normal(media, desv_tipica, [len(player_probability_goal)])

    player_probability_goal = player_probability_goal + noise

    for columna in range(len(player_probability_goal)):
        if player_probability_goal[columna] < 0.0:
            player_probability_goal[columna] = 0.0

    probabilities_gol = 0
    probabilities_goal = []

    for i in range(len(player_probability_goal)):
        probabilities_goal.append(0.0)
        probabilities_gol += player_probability_goal[i]

    # en otro for porque se necesita el valor entero de "probabilities"
    i = 0
    suma = 0.0
    for columna in range(len(player_probability_goal)):
        if probabilities_gol == 0:
            probabilities_goal[i] = 0.0
        else:
            probabilities_goal[i] = player_probability_goal[columna] / probabilities_gol
            suma = suma + probabilities_goal[i]
            i += 1

    if suma > 1:
        max_item = max(probabilities_goal, key=float)
        for j in range(len(probabilities_goal)):
            if probabilities_goal[j] == max_item:
                probabilities_goal[j] = probabilities_goal[j] - (suma - 1)
            j += 1
    elif suma < 1:
        if suma == 0.0:
            for j in range(len(probabilities_goal)):
                probabilities_goal[j] = float(1/len(probabilities_goal))
        else:
            max_item = max(probabilities_goal, key=float)
            for j in range(len(probabilities_goal)):
                if probabilities_goal[j] == max_item:
                    probabilities_goal[j] = probabilities_goal[j] + (1 - suma)
                j += 1

    scored = np.random.choice(player_probability_goal, p=probabilities_goal)

    if scored < 1.0:
        scored = 0
    else:
        scored = 1


    # - Probabilidad parada / fuera
    gk_probability_saved = np.array(gk_probability_saved, float)

    media = gk_probability_saved.mean()
    desv_tipica = gk_probability_saved.std()

    noise = np.random.normal(media, desv_tipica, [len(gk_probability_saved)])

    gk_probability_saved = gk_probability_saved + noise

    for columna in range(len(gk_probability_saved)):
        if gk_probability_saved[columna] < 0.0:
            gk_probability_saved[columna] = abs(gk_probability_saved[columna])

    probabilities_svd = 0
    probabilities_saved = []

    for i in range(len(gk_probability_saved)):
        probabilities_saved.append(0.0)
        probabilities_svd += gk_probability_saved[i]

    i = 0
    suma = 0.0
    for columna in range(len(gk_probability_saved)):
        if probabilities_svd == 0:
            probabilities_saved[i] = 0.0
        else:
            probabilities_saved[i] = gk_probability_saved[columna] / probabilities_svd
            suma = suma + probabilities_saved[i]
            i += 1

    if suma > 1:
        max_item = max(probabilities_saved, key=float)
        for j in range(len(probabilities_saved)):
            if probabilities_saved[j] == max_item:
                probabilities_saved[j] = probabilities_saved[j] - (suma - 1)
            j += 1
    elif suma < 1:
        if suma == 0.0:
            for j in range(len(probabilities_saved)):
                probabilities_saved[j] = float(1/len(probabilities_saved))
        else:
            max_item = max(probabilities_saved, key=float)
            for j in range(len(probabilities_saved)):
                if probabilities_saved[j] == max_item:
                    probabilities_saved[j] = probabilities_saved[j] + (1 - suma)
                j += 1

    saved = np.random.choice(gk_probability_saved, p=probabilities_saved)

    if scored >= 1 and saved >= 1:
        if scored > saved:
            scored = 1
            saved = 0
        else:
            scored = 0
            saved = 1
            if kick_direction != gk_direction:
                kick_direction = gk_direction # Normalmente suele ser más fallo del lanzador que acierto del potero, de ahí que modifico la dirección del lanzamiento y no al reves
    elif scored < 1 and saved < 1:
        scored = 0
        saved = 0
    elif scored >= 1 and saved < 1:
        scored = 1
        saved = 0
    elif scored < 1 and saved >= 1:
        scored = 0
        saved = 1
        if kick_direction != gk_direction:
            kick_direction = gk_direction  # Normalmente suele ser más fallo del lanzador que acierto del potero, de ahí que modifico la dirección del lanzamiento y no al reves

    df = pd.DataFrame(df)


    new_row = {'Season': season,
               'Match Week': match_week,
               'Date': pd.to_datetime('%d/%d/%d' % (day, month, year)),
               'Player': name_player,
               'Goalkeeper': name_goalkeeper,
               'Team_Player': local,
               'Team_Goalkeeper': visitor,
               'Match': '%s - %s' % (local, visitor),
               'Time of Penalty Awarded': time_penalty,
               'Final Results': '%d-%d' % (team_local, team_visitor),
               'Scored': scored,
               'Saved': saved,
               'Foot': foot,
               'Kick_Direction': kick_direction,
               'Keeper_Direction': gk_direction}

    # Añadimos la nueva fila al DataFrame
    df = df.append(new_row, ignore_index=True)


# Eliminamos las columnas que se crean como Unname
df.drop(df.filter(regex="Unname"), axis=1, inplace=True)

# Envíamos el DataFrame ya con la nueva línea al documento
df.to_excel('LaLiga_data_augmentation_gaussian_noise.xlsx', index=False)

print('Fin')