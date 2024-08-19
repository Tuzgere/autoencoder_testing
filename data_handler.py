import datetime
import os

import eco2ai
import numpy as np
import pandas as pd
import sklearn.model_selection
from tqdm import trange
import tensorflow as tf
import time
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


MACHINE_SENSORS = {
    "K-7502_ST-7501": [
        "75ti801",
        "75pi808",
        "75pi804",
        "75pi823",
        "75ti821",
        "75ti822",
        "75ti827",
        "75ti828",
        "75xi821bx",
        "75xi822by",
        "75ti824",
        "75zi800ba",
        "75zi800bb",
        "75si865r",
        "75pi870",
        "75pi168",
        "75ti829",
        "75ti830",
        "75ti831",
        "75ti832",
        "75ti836",
        "75ti834",
        "75xi823bx",
        "75xi824bx",
        "75zi801ba",
        "75zi801bb",
        "75fi034c",
        "75tt112",
        "75tt111",
        "75pt062",
        "75pi845",
        "75pdi853",
        "75PDIC858",
    ],
    "K-2201_KT-2201": [
        "22SI101",
        "22TI120",
        "22TI121",
        "22TI123",
        "22TI111",
        "22TI113",
        "22VI01",
        "22VI02",
        "22VI03",
        "22VI04",
        "22ZI09",
        "22ZI11",
        "22PI103",
        "22LI33",
        "22PI68",
        "22PI69",
        "22PI70",
        "22PI71",
        "22PDI74",
        "22PI95",
        "22PI65",
        "22TI65",
        "22PI106",
        "22TI115",
        "22TI117",
        "22TI118",
        "22TI119",
        "22VI05",
        "22VI06",
        "22VI07",
        "22VI08",
        "22ZI10",
        "22PIC44",
        "22PI101",
        "22TI42",
        "22TI40",
        "22FI09",
        "22AI402",
        "22TI99",
        "22LIC24",
        "22LI28",
        "22PI102",
        "22PI79",
        "22PDI80",
    ],
    "K-3201-A_KM-3201-A": [
        "32PI337",
        "32FIC21",
        "32FIC630",
        "32PI20",
        "32TI14",
        "32XI457",
        "32XI456",
        "32TI444",
        "32TI445",
        "32TI446",
        "32TI447",
        "32TI443",
        "32XI455",
        "32TI442",
        "32XI452",
        "32TI441",
        "32TI440",
        "32XI453",
        "32XI454",
        "32TI439",
        "32TI438",
        "32XI451",
        "32JI001A",
        "32TI448_8",
        "32TI448_7",
        "32TI448",
        "32TI448_2",
        "32TI448_3",
        "32TI448_4",
        "32TI448_5",
        "32TI448_6",
        "32PI424",
        "32TI430",
        "32PIC219",
    ],
    "K-3301-B_KT-3301-B": [
        "33FIC1",
        "33TI8",
        "33PIC19",
        "33TI17",
        "33TI11",
        "33FIC14",
        "33AI602",
        "33VI604",
        "33TI610",
        "33TI611",
        "33TI609",
        "33TI608",
        "33VI603",
        "33TI607",
        "33AI601",
        "33VI602",
        "33TI604",
        "33TI605",
        "33TI612",
        "33TI606",
        "33TI613",
        "33TI603",
        "33VI601",
        "33TI602",
        "33SI501A",
        "33PI222",
        "33TI230",
        "33PI601",
    ],
    "K-5701": [
        "57TI030",
        "57TI003",
        "57PIC004",
        "57PI005",
        "57FI001A",
        "57TI031",
        "57PI014",
    ],
}


def load_data(sensors: list) -> pd.DataFrame:
    df = pd.read_csv(
        f"{WORKING_DIR}/All_historical.csv",
        parse_dates=["Timestamp"],
        usecols=["Timestamp"] + sensors,
    )
    df.set_index("Timestamp", inplace=True)
    return df


def get_sequences(df: pd.DataFrame, window_size: int = 12):
    sequences_in = []
    sequences_out = []
    for i in trange(len(df) - window_size - 1):
        sequence_in = df.iloc[i : i + window_size]
        sequence_out = df.iloc[i + window_size + 1]
        sequences_in.append(sequence_in.values)
        sequences_out.append(sequence_out.values)
    return np.array(sequences_in), np.array(sequences_out)


def get_sensor_type(type="t"):
    sensor_type = []

    for _, sensors in MACHINE_SENSORS.items():
        for sensor in sensors:
            if type in sensor.lower():
                sensor_type.append(sensor)

    return sensor_type

def build_model(input_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(filters=32, kernel_size=6, strides=2, padding="causal", input_shape=(input_size, 1)),
            tf.keras.layers.ReLU(),

            tf.keras.layers.Conv1D(filters=64, kernel_size=4, strides=2, padding="causal"),
            tf.keras.layers.ReLU(),

            # tf.keras.layers.Conv1D(filters=64, kernel_size=7, padding="same"),
            # tf.keras.layers.ReLU(),

            # tf.keras.layers.ZeroPadding1D(padding=(int(np.ceil(input_size/2**3)), int(np.floor(input_size/2**3)))),
            tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=7, strides=2, padding="same"),
            tf.keras.layers.ReLU(),

            # tf.keras.layers.ZeroPadding1D(padding=(int(np.ceil(input_size/2**2)), int(np.floor(input_size/2**2)))),
            tf.keras.layers.Conv1DTranspose(filters=1, kernel_size=7, strides=2, padding="same"),
            tf.keras.layers.Activation('linear'),
        ]
    )
    return model


if __name__ == "__main__":
    sensors = get_sensor_type("t")
    sensors_data = load_data(sensors)
    seqs, _ = get_sequences(sensors_data)
    X, y = seqs, seqs
    # choose random column != 'Timestamp'
    random_sensors = np.random.choice(sensors_data.columns, replace=False)
    while random_sensors == 'Timestamp':
        random_sensors = np.random.choice(sensors_data.columns, replace=False)

    X1 = X[:, :, 42]
    print(X1.shape)
    y1 = X1

    print(f'X1, y1 for sensor {42}:')
    print(X1.shape, y1.shape)
    model = build_model(input_size=288)
    model.compile(optimizer="adam", loss="mse")
    for_c_names = pd.read_csv('emissions_autoencoder_train.csv').columns.tolist()
    for_c_names.append('mse')
    for_c_names.append('r2_score')
    for_c_names.append('test_on_num_samples')
    for_c_names.append('uuid')
    print(for_c_names)

    # tracker = eco2ai.Tracker(project_name="measurements_autoencoder", file_name='emissions_autoencoder_train.csv', cpu_processes='current', ignore_warnings=True, alpha_2_code='SI')
    # tracker.start()
    # model.fit(X1, y1, epochs=10)
    # model = tf.keras.models.load_model('autoencoder_model.keras')
    # tracker.stop()
    # model.save('autoencoder_model.keras')
    model = tf.keras.models.load_model('autoencoder_model.keras')
    tracker2 = eco2ai.Tracker(project_name="measurements_autoencoder", file_name='emissions_autoencoder_eval.csv',
                              cpu_processes='current', ignore_warnings=True, alpha_2_code='SI')

    collection_data = pd.DataFrame(columns=for_c_names)
    for i in range(16,1500):
        # take a random number of rows from X and y
        num_rows = np.random.randint(1000, X1.shape[0])
        X1_sample_train, X1_sample_test, y1_sample_train, y1_sample_test = sklearn.model_selection.train_test_split(
            X1,y1, test_size=num_rows)

        tracker2.start()
        y_pred = model.predict(X1_sample_test)
        y_pred = np.squeeze(y_pred, axis=-1)
        print(y_pred.shape, y1_sample_test.shape)

        tracker2.stop()
        data = pd.read_csv('emissions_autoencoder_eval.csv')
        os.remove('emissions_autoencoder_eval.csv')
        mse = sklearn.metrics.mean_squared_error(y1_sample_test, y_pred)
        r2_score = sklearn.metrics.r2_score(y1_sample_test, y_pred)
        data['mse'] = mse
        data['r2_score'] = r2_score
        data['test_on_num_samples'] = num_rows
        data['uuid'] = i
        # add data to collection_data
        collection_data = pd.concat([collection_data, data])
        print(f'mse for sensor 42 on test num{i}: {mse}    ||||||||||     r2_score: {r2_score}')

        tracker2 = eco2ai.Tracker(project_name="measurements_autoencoder",
                                  file_name='emissions_autoencoder_eval.csv', cpu_processes='current',
                                  ignore_warnings=True, alpha_2_code='SI')

        if datetime.datetime.now().hour == 9:
            print('saving data')
            collection_data.to_csv('emissions_autoencoder_eval_midway.csv', index=False)
            print('data saved')
            time.sleep(7200)






    collection_data.to_csv('emissions_autoencoder_eval_final.csv', index=False)

    print('done')
