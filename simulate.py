import json
from model import Model
from config import config
from absl import app
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from nltk import word_tokenize
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from math import sqrt
yf.pdr_override()  # <== that's all it takes :-)


portfolio = []
init_money = 30000
money = init_money
money_list = []


def sell(item, date_int, history_price, portfolio):
    global money
    if item not in portfolio:
        raise Exception('item not in portfolio')
    if item['type'] != 'buy':
        raise Exception('item type not buy')
    price = history_price[date_int][item['symbol']]
    print('selling', item['symbol'], date_int,
          price, item['price'], item['count'])
    money += price*item['count']
    portfolio.remove(item)


def sell_short(item, date_int, history_price, portfolio):
    global money
    if item not in portfolio:
        raise Exception('item not in portfolio')
    if item['type'] != 'buy_short':
        raise Exception('item type not buy short')
    price = history_price[date_int][item['symbol']]
    print('selling short', item['symbol'], date_int,
          price, item['price'], item['count'])
    money += price*item['count']
    portfolio.remove(item)


def buy(symbol, date_int, history_price, portfolio):
    global money
    price = history_price[date_int][symbol]
    count = 10000/price
    item = {
        'symbol': symbol,
        'date': date_int,
        'price': price,
        'count': count,
        'type': 'buy',
    }
    print('buying', symbol, price, count)
    money -= price*count
    portfolio.append(item)


def buy_short(symbol, date_int, history_price, portfolio):
    global money
    price = history_price[date_int][symbol]
    count = -10000/price
    item = {
        'symbol': symbol,
        'date': date_int,
        'price': price,
        'count': count,
        'type': 'buy_short',
    }
    print('buying short', symbol, price, count)
    money -= price*count
    portfolio.append(item)


def total(portfolio):
    result = 0
    for item in portfolio:
        result += item['price']*item['count']
    return result


def get_history_price(symbol):
    data = pdr.get_data_yahoo(symbol, start="2015-01-01", end="2018-01-01")
    data = data['Close'].to_dict()
    data = {k.strftime('%Y%m%d'): v for k, v in data.items()}
    return data


def save_history_price():
    symbol_list = json.load(
        open('/Users/cppowboy/data/senti-stock/symbol_list.json'))
    history_price = {}
    for symbol in symbol_list:
        if symbol == 'VRX' or symbol == 'MON':
            continue
        print('getting', symbol)
        history_price[symbol] = get_history_price(symbol)
    hp = {}
    for k1, vv in history_price.items():
        for k2, v in vv.items():
            if k2 not in hp:
                hp[k2] = {}
            hp[k2][k1] = v
    json.dump(hp, open('history_price.json', 'w'))


def get_news(filename):
    data = json.load(open(filename))
    result = {}
    for item in data:
        date_str = item['date_str']
        date_str = date_str[4:]+date_str[:4]
        if date_str not in result:
            result[date_str] = []
        result[date_str].append({
            'symbol': item['symbol'],
            'title': item['title']
        })
    return result


def load_model():
    device = torch.device(config.device)
    # init model
    data_dir = os.path.join(config.model, config.dataset)
    wordemb = np.loadtxt(os.path.join(data_dir, config.wordmat_file))
    targetemb = np.loadtxt(os.path.join(data_dir, config.targetmat_file))
    model = Model(kernel_size=config.kernel_size, num_channel=config.num_channel, dim_middle=config.dim_middle,
                  num_concept=config.num_concept, dim_concept=config.dim_concept, num_classification=config.num_class,
                  maxlen=config.sent_limit, dim_word=config.dim_word, dropout_rate=config.dropout_rate, wordemb=wordemb, targetemb=targetemb)
    # load model
    model_save_dir = os.path.join(
        config.model_save, config.dataset, config.model)
    if 'cpu' in config.device:
        model.load_state_dict(torch.load(os.path.join(
            model_save_dir, 'best.pth'), map_location='cpu'))
    else:
        model.load_state_dict(torch.load(
            os.path.join(model_save_dir, 'best.pth')))
    model = model.to(device)
    model.eval()
    return model


def data2feature(sentence, symbol, word2id, target2id):
    sentence, symbol = sentence.lower().strip(), symbol.lower().strip()
    sent_ids = np.zeros([config.sent_limit])
    words = word_tokenize(sentence)
    length = len(words)
    for i, t in enumerate(words):
        if t in word2id:
            sent_ids[i] = word2id[t]
        else:
            sent_ids[i] = word2id['<UNKNOWN>']
    aspect_id = target2id[symbol]
    return sent_ids.reshape(1, -1), np.array([[aspect_id]]), np.array([[length]])


def predict(model, news_list, word2id, target2id):
    result = {
        'increase': [],
        'decrease': [],
        'other': []
    }
    for news in news_list:
        symbol = news['symbol']
        title = news['title']
        sent_ids, aspect_ids, lens = data2feature(
            title, symbol, word2id, target2id)
        sent_ids, aspect_ids, lens = torch.from_numpy(sent_ids).long().to(config.device), torch.from_numpy(
            aspect_ids).long().to(config.device), torch.from_numpy(lens).long().to(config.device)
        pred = model(sent_ids, aspect_ids, lens)
        pred = pred.data.cpu().numpy()[0]
        pred = np.argmax(pred)
        if symbol == 'VRX' or symbol == 'MON':
            continue
        if pred == 0:
            result['decrease'].append(symbol)
        elif pred == 1:
            result['other'].append(symbol)
        elif pred == 2:
            result['increase'].append(symbol)
        return result


def nextdate(date_str, term):
    date = datetime.strptime(date_str, '%Y%m%d')
    nextdate = date + timedelta(days=term)
    return nextdate.strftime('%Y%m%d')


def annulized_returns(money_list):
    n = len(money_list)
    p_start = money_list[0]
    p_end = money_list[-1]
    pr = (p_end/p_start)**(250/n)-1
    return pr


def average(a_list):
    return sum(a_list)/len(a_list)


def daily_return(ab_list):
    a_list, b_list = ab_list[:-1], ab_list[1:]
    return [b/a-1 for a, b in zip(a_list, b_list)]


def volatility(a_list):
    a_mean = average(a_list)
    total = 0
    for a in a_list:
        total += (a-a_mean)**2
    return sqrt(total * 250 / (len(a_list)-1))


def var(a_list):
    a_mean = average(a_list)
    total = 0
    for a in a_list:
        total += (a-a_mean)*(a-a_mean)
    return sqrt(total / (len(a_list)-1))


def beta(p_list, b_list):
    c = np.cov(np.array(p_list), np.array(b_list))
    return c[0][1] / c[1][1]


def alpha(p_list, b_list, rf):
    pr = annulized_returns(p_list)
    br = annulized_returns(b_list)
    b = beta(p_list, b_list)
    return pr-rf-b*(br-rf)


def sharp(p_list, rf):
    pr = annulized_returns(p_list)
    sigma_p = volatility(daily_return(p_list))
    return (pr-rf)/sigma_p


def information_ratio(p_list, b_list):
    pr = annulized_returns(p_list)
    br = annulized_returns(b_list)
    p_list, b_list = daily_return(p_list), daily_return(b_list)
    sigma_t = volatility([p-b for p, b in zip(p_list, b_list)])
    return (pr-br)/sigma_t


def maxdrawdown(p_list):
    cur_max = -10000
    res = 0
    for p in p_list:
        if p < cur_max:
            r = (cur_max-p)/cur_max
            res = max(res, r)
        cur_max = max(cur_max, p)
    return res


def get_benchmark(money_list):
    sp500 = get_history_price('^GSPC')
    json.dump(sp500, open('sp500.json', 'w'))
    res = []
    for item in money_list:
        res.append(sp500[item[0]])
    return res


def evaluate(p_list, b_list, rf):
    score = {
        'pr': annulized_returns(p_list),
        'br': annulized_returns(b_list),
        'beta': beta(p_list, b_list),
        'alpha': alpha(p_list, b_list, rf),
        'volatility': volatility(daily_return(p_list)),
        'sharp ratio': sharp(p_list, rf),
        'information ratio': information_ratio(p_list, b_list),
        'max drawdown': maxdrawdown(p_list)
    }
    return score


def draw_money_list(p_list, b_list):
    plt.plot(np.array(p_list),label='strategy')
    plt.plot(np.array(b_list),label='benchmark')
    plt.xlabel('trading day')
    plt.ylabel('total assets ($)')
    plt.legend()
    plt.show()


def main(_):
    # save_history_price()
    term = 1
    history_price = json.load(open('history_price.json'))
    news = get_news('/Users/cppowboy/data/senti-stock/middle.json')
    model = load_model()
    data_dir = os.path.join(config.model, config.dataset)
    word2id = json.load(open(os.path.join(data_dir, 'word2id.json')))
    target2id = json.load(open(os.path.join(data_dir, 'target2id.json')))
    date_list = list(set(history_price.keys()) & set(news.keys()))
    date_list = sorted(date_list)
    for date_str in date_list:
        print('=='*10, date_str, '=='*10)
        # process portfolio
        for item in portfolio:
            nextdate_str = nextdate(item['date'], term)
            if date_str not in history_price or item['symbol'] not in history_price[date_str]:
                continue
            if item['type'] == 'buy':
                price = history_price[date_str][item['symbol']]
                if price / item['price'] > 1.01:
                    sell(item, date_str, history_price, portfolio)
                elif nextdate_str >= date_str:
                    sell(item, date_str, history_price, portfolio)
            elif item['type'] == 'buy_short':
                price = history_price[date_str][item['symbol']]
                if price / item['price'] < 0.99:
                    sell_short(item, date_str, history_price, portfolio)
                elif nextdate_str >= date_str:
                    sell_short(item, date_str, history_price, portfolio)
            else:
                raise Exception('unknown item type')
        # predict
        if date_str in news:
            pred = predict(model, news[date_str], word2id, target2id)
            print(pred)
            for symbol in pred['increase']:
                buy(symbol, date_str, history_price, portfolio)
            for symbol in pred['decrease']:
                buy_short(symbol, date_str, history_price, portfolio)
        # count money
        t = total(portfolio)
        money_list.append([date_str, money, t, money+t])
    print(money_list)
    date_list, _, _, p_list = zip(*money_list)
    b_list = get_benchmark(money_list)
    b_start = b_list[0]
    b_list = [b/b_start*init_money for b in b_list]
    print(average(p_list), average(b_list))
    score = evaluate(p_list, b_list, 0.03)
    print(score)
    draw_money_list(p_list, b_list)


if __name__ == '__main__':
    app.run(main)
