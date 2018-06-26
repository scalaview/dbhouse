import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def prepare_train_data(syml):
    import models
    hist = pd.read_sql_query('SELECT `date`, close_price, id, tag FROM daily_prices WHERE fsymbol="'+syml+'" AND tsymbol="USDT" AND date between "2018-01-01" and "2018-06-24" ', models.engine)
    hist = hist.set_index('date')
    hist.index = pd.to_datetime(hist.index, unit='s')
    return hist.fillna(0)
# ALTER TABLE daily_prices ADD tag int(11);
# day 3
def sort_tag(syml, n_day=2):
    from models import DailyPrice
    import models
    connection = models.engine.connect()
    data = prepare_train_data(syml)
    data['close_price_diff_'+str(n_day)] = data['close_price'].diff(n_day).shift(-n_day) / data['close_price'] * 100
    length = len(data)
    for (line_number, (index, d)) in enumerate(data.iterrows()):
        if np.isnan(d['close_price_diff_'+str(n_day)]):
            continue
        updated = False
        if line_number+7 < length:
            max_price = np.max(d['close_price'])
            min_price = np.min(d['close_price'])
            if (max_price - d['close_price']) / d['close_price'] * 100 <= 0.50:
                u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(tag=1)
                updated = True
            elif (d['close_price'] - min_price) / d['close_price'] * 100 <= 0.50:
                u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(tag=2)
                updated = True

        if not updated and d['close_price_diff_'+str(n_day)] >= 5.00:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(tag=1)
        elif not updated and d['close_price_diff_'+str(n_day)] > 0.00:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(tag=2)
        elif not updated and d['close_price_diff_'+str(n_day)] <= -2.00:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(tag=3)
        elif not updated and d['close_price_diff_'+str(n_day)] <= 0.00:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(tag=4)
        elif not updated:
            u = DailyPrice.__table__.update().where(DailyPrice.id==d["id"]).values(tag=5)
        connection.execute(u)
    connection.close()


def paint_plot_with_color(datelist, closeidx, latent_states_sequence):
    for i in range(0, 6):
        state = (latent_states_sequence == i)
        plt.plot(datelist[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
        plt.legend()
        plt.grid(1)
    plt.show()

if __name__ == '__main__':
    sort_tag('btc')
    data = prepare_train_data('btc')
    paint_plot_with_color(data.index, data['close_price'].values, data['tag'])
