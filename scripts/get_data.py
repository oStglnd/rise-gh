
import pandas as pd

def getData(path, cats, names, drop_dates):
    data = {}
    for cat in cats:
        file = path + cat + '.csv'
        dataset = pd.read_csv(file)
        
        cols = dataset.columns.values.tolist()

        for name in names:
            dtypes = [col for col in cols if name in col]
            
            for dtype in dtypes:
                #typ = dtype.split('__')[-1]
                data[(cat, name)] = dataset[dtype].values

    dates = dataset.Timestamps.apply(
        lambda date: pd.to_datetime(date, unit='s')
    ).rename('date')

    # create dataFrame w. dict and dates as index
    data = pd.DataFrame(
        index=dates,
        data=data
    )

    # SORT columns in data 
    data = data.reindex(
        sorted(data.columns,
               key=lambda c: c[0]),
        axis=1
    )

    # REMOVE NANS
    data['date'] = data.index
    data = data[data.date.apply(lambda d: d.date().isoformat() not in drop_dates)]
    del data['date']
    
    return data