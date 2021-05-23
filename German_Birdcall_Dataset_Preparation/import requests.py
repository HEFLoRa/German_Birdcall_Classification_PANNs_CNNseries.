import requests
import os
import pandas as pd
import json
import re

url = 'https://www.xeno-canto.org/sounds/uploaded/HEYJSRUDZZ/XC127145-Rhea_20130216_142948_prepared.mp3'

def get_list_of_birds(url, pages):

    rec_dict = {}

    for page in range(1, pages+1):

        #url = 'https://www.xeno-canto.org/sounds/uploaded/HEYJSRUDZZ/XC127145-Rhea_20130216_142948_prepared.mp3'
        r = requests.get(url+"&page={0}".format(page), allow_redirects=True)
        converted = remove_non_ascii(r.text)
        contant = json.loads(converted)


        for key in contant['recordings'][0].keys():

            liste = []
            for entry in contant['recordings']:

                liste.append(entry[key])
                

            if(key in rec_dict.keys()):
                rec_dict[key].extend(liste)
            else:
                rec_dict[key] = liste    
        
    #liste = [pd.DataFrame(entry)]
    #liste.append(pd.DataFrame(entry))

    result = pd.DataFrame(rec_dict)
    return result

def remove_non_ascii(text):

    reaesc_compiler = re.compile(r'\x1b[^m]*m')
    device_info = reaesc_compiler.sub('', text)
    device_info = re.sub('\t', ' ', device_info)      
    device_info = re.sub('\n', ' ', device_info)   
    device_info = re.sub('<nil>', ' ', device_info)   

    return device_info

def download_sound_files(dataframe, bird_name, sub_species):

    print(dataframe.head())
    bird_dir = bird_name
    if(sub_species):
        path= os.path.join(os.getcwd(), bird_dir, sub_species)
    else:
        path= os.path.join(os.getcwd(), bird_dir)

    if(bird_dir not in os.listdir(os.getcwd())):

        try:
            os.mkdir(os.path.join(os.getcwd(), bird_dir))
        except OSError:
            print ("Creation of the directory %s failed" % bird_dir)
        else:
            print ("Successfully created the directory %s " % bird_dir)
    else:
        print ("Directory %s already exists" % bird_dir)

    if(sub_species not in os.listdir(os.path.join(os.getcwd(), bird_dir))):

        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the Sub-directory %s failed" % path)
        else:
            print ("Successfully created the Sub-directory %s " % path)
    else:
        print ("Sub-Directory %s already exists" % path)


    file_names = dataframe['file-name'].tolist()
    for idx, url in enumerate(dataframe['file']):

        file_name = file_names[idx]

        if(file_name not in os.listdir(path)):
            try:
                r = requests.get("http:"+url, allow_redirects=True)
                open(os.path.join(path, file_name), 'wb').write(r.content)
            except:
                print("Connection Error")

    


if __name__ == "__main__":

    url = 'https://www.xeno-canto.org/sounds/uploaded/HEYJSRUDZZ/XC127145-Rhea_20130216_142948_prepared.mp3'
    url_req = 'https://www.xeno-canto.org/api/2/recordings?query=cnt:Germany+gen:"Turdus"'
    url_req = 'https://www.xeno-canto.org/api/2/recordings?query=cnt:Germany'

    birds_df = get_list_of_birds(url_req, pages=44)
    birds_df.to_csv("xeno_canto_sounds_germany.csv")
    print(birds_df.head())

    bird_names = birds_df.gen.unique()
    print(bird_names)
    for idx, name in enumerate(bird_names):
        name_filtered_df = birds_df.loc[birds_df['gen'] == name]

        sub_species = name_filtered_df.sp.unique()

        for species in sub_species:
            filter_df = name_filtered_df.loc[birds_df['sp'] == species]
            download_sound_files(filter_df, name, species)
        print("#####################################################################")
        print("######--------{0} % of all bird sounds are downloaded.-------########".format(round((idx+1)/len(bird_names),2)*100))
        print("#####################################################################")

"""
        if(len(sub_species)==1):
            filter_df = birds_df.loc[birds_df['gen'] == name]
            download_sound_files(filter_df, name, sub_species=None)
        else:
            """
