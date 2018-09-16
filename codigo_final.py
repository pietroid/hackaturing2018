import pandas as pd
import datetime
import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

dataset = pd.read_csv('../hackaturing2018_dados/hackaturing_table.dsv', sep='|')

hosp=dataset[dataset['base_hackaturing.cnpj']=='1deff610d3afbb1bc2403d03bb1e53ca']
hosp[~hosp['base_hackaturing.cbos_solicitante'].isnull()]
hosp_ie=hosp[hosp['base_hackaturing.carater_atendimento']=='ELETIVO']
hosp_ie=hosp_ie[hosp_ie['base_hackaturing.tipo_guia']=='Internacao']
hosp_f=hosp_ie[['base_hackaturing.id_beneficiario','base_hackaturing.data_nascimento','base_hackaturing.data_entrada','base_hackaturing.data_saida','base_hackaturing.data_item','base_hackaturing.tipo_item','base_hackaturing.servico','base_hackaturing.descricao_despesa','base_hackaturing.quantidade', 'base_hackaturing.valor_item','base_hackaturing.valor_cobrado','base_hackaturing.valor_pago']]
hosp_f=hosp_f.sort_values('base_hackaturing.id_beneficiario')

now=pd.Timestamp(datetime.datetime.now())
hosp_f['base_hackaturing.data_nascimento'] = pd.to_datetime(hosp_f['base_hackaturing.data_nascimento'], format='%Y-%m-%d')    # 1
hosp_f['age'] = (now - hosp_f['base_hackaturing.data_nascimento']).astype('<m8[Y]')
hosp_f = hosp_f.drop('base_hackaturing.data_nascimento', 1)

hosp_f['base_hackaturing.data_entrada'] = pd.to_datetime(hosp_f['base_hackaturing.data_entrada'], format='%Y-%m-%d',unit='s')


hosp_f['base_hackaturing.data_saida'] = pd.to_datetime(hosp_f['base_hackaturing.data_saida'], format='%Y-%m-%d')    # 1
hosp_f['base_hackaturing.data_entrada'] = pd.to_datetime(hosp_f['base_hackaturing.data_entrada'], format='%Y-%m-%d')    # 1

hosp_f['intern_time'] = (hosp_f['base_hackaturing.data_saida'] - hosp_f['base_hackaturing.data_entrada'])
hosp_f = hosp_f.drop('base_hackaturing.data_saida', 1)

itens = ['PROCEDIMENTO', 'MATERIAIS', 'MEDICAMENTOS', 'TAXAS DIVERSAS', 'DIARIAS', 'GASES MEDICINAIS', 'OPME']
for item in itens:
    hosp_f[item] = hosp_f['base_hackaturing.tipo_item'] == item
    
hosp_f['base_hackaturing.data_item'] = pd.to_datetime(hosp_f['base_hackaturing.data_item'], format='%Y-%m-%d')    # 1
hosp_f['base_hackaturing.data_entrada'] = pd.to_datetime(hosp_f['base_hackaturing.data_entrada'], format='%Y-%m-%d')    # 1

hosp_f['aplication_time'] = (hosp_f['base_hackaturing.data_item'] - hosp_f['base_hackaturing.data_entrada'])
hosp_f = hosp_f.drop('base_hackaturing.data_item', 1)

hosp_f = hosp_f.drop('base_hackaturing.tipo_item', 1)
hosp_f = hosp_f.drop('base_hackaturing.descricao_despesa', 1)

a =hosp_f['base_hackaturing.servico'].value_counts()
i=1
for item in a.keys()[20:40]:
    hosp_f[i*10] = hosp_f['base_hackaturing.servico'] == item
    i+=1

hosp_f=hosp_f.reset_index(drop=True)

a =hosp_f['base_hackaturing.servico'].value_counts()
i=1
for item in a.keys()[0:20]:
    hosp_f[i*10] = hosp_f['base_hackaturing.servico'] == item
    i+=1

#Levantamento das metricas
vector_gastro=[31001076,31002218,31002390,84430095,84430109,84430460,84430540,84430478]
hosp_input = []
hosp_output = []
lastBen=hosp_f['base_hackaturing.id_beneficiario'][0]
lastTimeEntrada=hosp_f['base_hackaturing.data_entrada'][0].timestamp()
age=0
total_intern_time=0
partial_intern_time=0
valor_item=0
procedimento=0
materiais=0
diarias=0
taxas_diversas=0
gases_medicinais=0
opme=0
servs=20*[0]
gastro=False
for i in range(0,len(hosp_f['base_hackaturing.id_beneficiario'])):
    if(hosp_f['base_hackaturing.data_entrada'][i].timestamp()!=lastTimeEntrada):
        lastTimeEntrada=hosp_f['base_hackaturing.data_entrada'][i].timestamp()
        total_intern_time+=partial_intern_time
        partial_intern_time=0 
    if(hosp_f['base_hackaturing.id_beneficiario'][i]!=lastBen):
        hosp_input.append([age,total_intern_time,valor_item,procedimento,materiais,diarias,taxas_diversas,gases_medicinais,opme]+servs)
        hosp_output.append(gastro)
        lastBen=hosp_f['base_hackaturing.id_beneficiario'][i]
        age=0
        gastro=False
        total_intern_time=0
        valor_item=0
        procedimento=0
        materiais=0
        diarias=0
        taxas_diversas=0
        gases_medicinais=0
        opme=0
        servs=20*[0]
    if(hosp_f['base_hackaturing.servico'][i] in vector_gastro):
        gastro=True
    age=hosp_f['age'][i]
    partial_intern_time=hosp_f['intern_time'][i].days
    valor_item+=hosp_f['base_hackaturing.valor_item'][i]
    procedimento+=hosp_f['PROCEDIMENTO'][i]
    materiais+=hosp_f['MATERIAIS'][i]
    diarias+=hosp_f['DIARIAS'][i]
    taxas_diversas+=hosp_f['TAXAS DIVERSAS'][i]
    gases_medicinais+=hosp_f['GASES MEDICINAIS'][i]
    opme+=hosp_f['OPME'][i]
    for j in range(0,len(servs)):
        servs[j]+=hosp_f['base_hackaturing.quantidade'][i]*hosp_f[(j+1)*10][i]
total_intern_time+=partial_intern_time
hosp_input.append([age,total_intern_time,valor_item,procedimento,materiais,diarias,taxas_diversas,gases_medicinais,opme]+servs)
hosp_output.append(gastro)

#random forest
clf = RandomForestClassifier(max_depth=10, random_state=0)
hosp_input=np.array(hosp_input)
hosp_output=np.array(hosp_output)
hosp_input[np.isnan(hosp_input)]=0
hosp_output[np.isnan(hosp_output)]=0
clf.fit(hosp_input, hosp_output)

##Printa Importancia das entradas
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
print(clf.feature_importances_)

inp=hosp_input
outp=hosp_output

#Downsampling
for i in range(0,1000):
    if(outp[i]==False):
        del outp[i]
        del inp[i]
#Upsampling
for j in range(0,5):
	for i in range(0,2000):
	    if(outp[i]==True):
	        outp.append(outp[i])
	        inp.append(inp[i])

#neural network com os principais
inp=inp[:,range(3,4,7,9,12)]
model = Sequential()
model.add(Dense(10, input_dim=26, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='random_uniform', activation='sigmoid'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(inp,outp,epochs=100,batch_size=5,verbose=1,validation_split = 0.2)

model.save('modelo_final.h5')