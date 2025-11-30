import os
import glob

# znajdź wszystkie pliki kończące się na "_d2" w bieżącym folderze
files_to_delete = glob.glob("*_d2.*")

# usuń je
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Usunięto: {file_path}")
    except Exception as e:
        print(f"Błąd przy usuwaniu {file_path}: {e}")
