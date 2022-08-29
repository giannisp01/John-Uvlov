import stock_info as si
import csv
counter=0
with open('company_tickers.csv', mode ='r')as file:
    csvFile = csv.reader(file)
    for lines in csvFile:

        print(f"Company {counter} : {lines[2]} ", end=" ")
        data = si.get_earnings_history(lines[2])
        print(f"{lines[2]} length data : {len(data)}")
        counter+=1