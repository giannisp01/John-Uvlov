import stock_info as si
import csv
from requests import get

ip = get('https://api.ipify.org').text
alldata = []
data = si.get_earnings_history('MSFT')
print(f'My public IP address is: {ip}')
counter = 0
#with open('./Github/John-Uvlov/Yahoo Earnings/Yahoo_Fin/yahoo_fin_fin/yahoo_fin/company_tickers.csv', mode='r') as file:
with open('company_tickers.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        data = si.get_earnings_history(lines[2])
        alldata.append(data)
        if counter%25==0 and counter!=0:
            import time
            print("Sleeping")
            time.sleep(120)
            print(counter+1)
        counter += 1


