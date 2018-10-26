import vis.core
import proc.readdata

if __name__ == "__main__":
    cabs_file = r"C:\Users\Jason\Desktop\BDownloads\cabspottingdata\_cabs.txt"
    cablist_reader = proc.readdata.CabListReader(cabs_file)
    cablist_reader.read_data()