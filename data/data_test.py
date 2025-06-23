import polars as pl

file_path="../data/archive/all_songs_data.csv"
songs_data = pl.read_csv(file_path, encoding='ISO-8859-1').with_row_index(offset=1)

print(songs_data[0]["Lyrics"])

lyrics = songs_data[0]["Lyrics"].item()
print(lyrics)