import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def prepare_train_data(syml):
    import models
    hist = pd.read_sql_query('SELECT `date`, close_price, id, trend_tag, point_tag FROM daily_prices WHERE fsymbol="'+syml+'" AND tsymbol="USDT"', models.engine)
    hist = hist.set_index('date')
    hist.index = pd.to_datetime(hist.index, unit='s')
    return hist.fillna(0)

def find_points(syml):
    from models import DailyPrice
    import models
    connection = models.engine.connect()
    data = prepare_train_data(syml)
    length = len(data)
    for (line_number, (index, d)) in enumerate(data.iterrows()):
        if np.isnan(d['close_price']):
            continue
        updated = False
        if line_number+7 < length and line_number-7 >= 0:
            max_price = np.max(data['close_price'][line_number-7:line_number+7])
            min_price = np.min(data['close_price'][line_number-7:line_number+7])
            u = None
            if (max_price - d['close_price']) / d['close_price'] * 100 <= 0.50:
                u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(point_tag=1)
                updated = True
            elif (d['close_price'] - min_price) / d['close_price'] * 100 <= 0.50:
                u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(point_tag=2)
                updated = True
            if u is not None:
                connection.execute(u)
    connection.close()
    return range(0, 3)

# ALTER TABLE daily_prices ADD tag int(11);
# day 3
def sort_tag(syml, n_day=2):
    from models import DailyPrice
    import models
    connection = models.engine.connect()
    data = prepare_train_data(syml)
    data['close_price_diff_'+str(n_day)] = data['close_price'].diff(n_day).shift(-n_day) / data['close_price'] * 100
    length = len(data)
    for index, d in data.iterrows():
        if np.isnan(d['close_price_diff_'+str(n_day)]):
            continue
        if d['close_price_diff_'+str(n_day)] >= 5.00:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(trend_tag=1)
        elif d['close_price_diff_'+str(n_day)] > 0.00:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(trend_tag=2)
        elif d['close_price_diff_'+str(n_day)] <= -2.00:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(trend_tag=3)
        elif d['close_price_diff_'+str(n_day)] <= 0.00:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(trend_tag=4)
        else:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(trend_tag=5)
        connection.execute(u)
    connection.close()
    return range(0, 6)


def paint_plot_with_color(datelist, closeidx, latent_states_sequence, _range=(0, 3)):
    for i in _range:
        state = (latent_states_sequence == i)
        plt.plot(datelist[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
        plt.legend()
        plt.grid(1)
    plt.show()

if __name__ == '__main__':
    _range = find_points('btc')
    _range = sort_tag('btc')
    data = prepare_train_data('btc')
    paint_plot_with_color(data.index, data['close_price'].values, data['trend_tag'], range(0, 6))
