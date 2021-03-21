import csv

with open('gravacoes/video2fr.csv') as csv_file:

    with open('gravacoes/video2fr_teste.csv', 'w', newline='') as file:

        writer = csv.writer(file)

        csv_reader = csv.DictReader(csv_file, fieldnames=["d", "a", "f", "t"])
        csv_reader.__next__()

        for row in csv_reader:
            valor = row["t"].replace(']','') + ', ' + row["d"].replace('[','') + ', ' + row["a"]
            print(valor)

            writer.writerow([row["t"].replace(']',''),row["d"].replace('[',''),row["a"]])
        
            