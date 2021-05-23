import os, sys
import pandas as pd
train_csv = pd.DataFrame(columns = ["gen","sp","filename"])
path = "/mnt/Germany_Birdcall/Germany_Birdcall_resampled"
primary_labels = os.listdir( path )
i = 0
BIRD_CODE = {}
for gen in primary_labels:
  BIRD_CODE[gen] = i
  i = i + 1
print(BIRD_CODE)


'''Running result: '''
BIRD_CODE = {
    'Pandion': 0, 'Iduna': 1, 'Lyrurus': 2, 'Aquila': 3, 'Garrulus': 4, 'Circus': 5,
    'Sylvia': 6, 'Coccothraustes': 7, 'Clangula': 8, 'Pernis': 9, 'Nymphicus': 10,
    'Burhinus': 11, 'Chroicocephalus': 12, 'Pluvialis': 13, 'Ixobrychus': 14, 'Columba': 15,
    'Metrioptera': 16, 'Somateria': 17, 'Plectrophenax': 18, 'Stethophyma': 19, 'Psittacula': 20,
    'Tetrao': 21, 'Glaucidium': 22, 'Jynx': 23, 'Crex': 24, 'Conocephalus': 25,
    'Gryllus': 26, 'Himantopus': 27, 'Remiz': 28, 'Aix': 29, 'Linaria': 30,
    'Locustella': 31, 'Calidris': 32, 'Aythya': 33, 'Cuculus': 34, 'Lullula': 35,
    'Recurvirostra': 36, 'Spatula': 37, 'Thalasseus': 38, 'Chlidonias': 39, 'Tetrastes': 40,
    'Ardea': 41, 'Phylloscopus': 42, 'Larus': 43, 'Streptopelia': 44, 'Limosa': 45,
    'Lanius': 46, 'Oecanthus': 47, 'Scolopax': 48, 'Platalea': 49, 'Phalacrocorax': 50,
    'Tarsiger': 51, 'Perdix': 52, 'Eremophila': 53, 'Caprimulgus': 54, 'Ciconia': 55,
    'Hydroprogne': 56, 'Grus': 57, 'Otis': 58, 'Botaurus': 59, 'Gallinago': 60,
    'Picoides': 61, 'Cyanistes': 62, 'Dendrocoptes': 63, 'Hirundo': 64, 'Chloris': 65,
    'Galerida': 66, 'Tadorna': 67, 'Tachymarptis': 68, 'Delichon': 69, 'Strix': 70,
    'Oriolus': 71, 'Pica': 72, 'Haliaeetus': 73, 'Sonus': 74, 'Chorthippus': 75,
    'Numenius': 76, 'Lagopus': 77, 'Ichthyaetus': 78, 'Porzana': 79, 'Passer': 80,
    'Rallus': 81, 'Ardeola': 82, 'Spinus': 83, 'Poecile': 84, 'Dryocopus': 85,
    'Sturnus': 86, 'Bubo': 87, 'Riparia': 88, 'Gavia': 89, 'Upupa': 90,
    'Calcarius': 91, 'Gallinula': 92, 'Otus': 93, 'Dendrocopos': 94, 'Loxia': 95,
    'Podiceps': 96, 'Uria': 97, 'Dryobates': 98, 'Alauda': 99, 'Prunella': 100,
    'Anthus': 101, 'Panurus': 102, 'Fulica': 103, 'Rhea': 104, 'Netta': 105,
    'Carduelis': 106, 'Cisticola': 107, 'Charadrius': 108, 'Amazona': 109, 'Apus': 110,
    'Haematopus': 111, 'Carpodacus': 112, 'Rissa': 113, 'Mystery': 114, 'Nemobius': 115,
    'Alopochen': 116, 'Morus': 117, 'Falco': 118, 'Fringilla': 119, 'Anser': 120,
    'Mareca': 121, 'Actitis': 122, 'Motacilla': 123, 'Alcedo': 124, 'Mergus': 125,
    'Arenaria': 126, 'Tachybaptus': 127, 'Corvus': 128, 'Vanellus': 129, 'Calandrella': 130,
    'Pyrrhocorax': 131, 'Ficedula': 132, 'Cygnus': 133, 'Saxicola': 134, 'Bucephala': 135,
    'Bombycilla': 136, 'Hippolais': 137, 'Branta': 138, 'Sitta': 139, 'Emberiza': 140,
    'Regulus': 141, 'Tyto': 142, 'Pseudochorthippus': 143, 'Gryllotalpa': 144, 'Muscicapa': 145,
    'Phoenicopterus': 146, 'Phaneroptera': 147, 'Luscinia': 148, 'Lophophanes': 149, 'Erithacus': 150,
    'Phasianus': 151, 'Milvus': 152, 'Anas': 153, 'Acanthis': 154, 'Picus': 155,
    'Aegolius': 156, 'Certhia': 157, 'Gelochelidon': 158, 'Periparus': 159, 'Melanitta': 160,
    'Aegithalos': 161, 'Sterna': 162, 'Buteo': 163, 'Serinus': 164, 'Phoenicurus': 165,
    'Cinclus': 166, 'Stercorarius': 167, 'Turdus': 168, 'Troglodytes': 169, 'Athene': 170,
    'Montifringilla': 171, 'Tringa': 172, 'Coturnix': 173, 'Merops': 174, 'Oenanthe': 175,
    'Coloeus': 176, 'Accipiter': 177, 'Pyrrhula': 178, 'Cettia': 179, 'Nucifraga': 180,
    'Nycticorax': 181, 'Asio': 182, 'Acrocephalus': 183, 'Clanga': 184, 'Parus': 185
}
