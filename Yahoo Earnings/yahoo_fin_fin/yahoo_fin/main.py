import stock_info as si
import csv
from requests import get

ip = get('https://api.ipify.org').text

data = si.get_earnings_history('MSFT')
print(f'My public IP address is: {ip}')
counter = 0
with open('company_tickers.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        data = si.get_earnings_history(lines[2])
        if counter%25==0 and counter!=0:
            import time
            print("Sleeping")
            time.sleep(120)
        counter += 1


