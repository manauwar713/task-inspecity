import time
import pickle
time1 = time.time()
time8 = time.time()
import subprocess
time9 = time.time()
print(time9-time8)
time10 = time.time()

import csv 

time11 = time.time()
print(time11-time10)

time14 = time.time()
import multiprocessing
time15 = time.time()
print(time15-time14)

time6 = time.time()
import numpy as np
time7 = time.time()
from functools import lru_cache
print(time7-time6)
time2 = time.time()


time3 = time.time()


sext = 'source-extractor ' + '/home/alam/task/task-inspecity/source-ex/img_fits.fits'

subprocess.run(sext, shell=True,capture_output=True, text=True)
# Read data from the sextractor format file
with open('/home/alam/task/task-inspecity/source-ex/test.cat', 'r') as infile:
    # Assuming the header is present, skip it
    next(infile)
    next(infile)
    next(infile)
    
    # Read the data into a list of dictionaries
    sex_data = [line.split() for line in infile]

# Convert string values to float
for i in range(len(sex_data)):
    sex_data[i] = [float(value) for value in sex_data[i]]

# Sort the data based on 'MAG_ISO'
sorted_sex_data = sorted(sex_data, key=lambda x: x[2])

# Take the first 40 elements
selected_sex_data = sorted_sex_data[:40]

# Extract columns for 'X_IMAGE', 'Y_IMAGE', and 'MAG_ISO'
sex_x1 = [(entry[0] - 512) * 0.00270 for entry in selected_sex_data]
sex_y1 = [(entry[1] - 512) * 0.00270 for entry in selected_sex_data]
sex_mag = [entry[2] for entry in selected_sex_data]

# Write data to a new CSV file
output_path = 'sext_t.csv'
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    
    # Write the data directly
    for x, y, mag in zip(sex_x1, sex_y1, sex_mag):
        writer.writerow([x, y, mag])


time4 = time.time()

parameters1 = 'trirad=0.002 nobj=15 max_iter=1 matchrad=1 scale=1'




ra_dec = [(ra, dec) for ra in range(0, 360, 20) for dec in range(-80, 90, 20)]


@lru_cache
def call_match(ra_dec):
    RA1, DEC1 = ra_dec
    # Transform RA and DEC in string and make the path for catalog.
    path_catalog1 = f'/home/alam/task/task-inspecity/notebooks/Catalog/Projected/cat_RA_'
    path_catalog2 = str(RA1) + '_DEC_' + str(DEC1)
    path_catalog3 = path_catalog1 + path_catalog2
    # Do Match.
    file_path = '/home/alam/task/sext_t.csv'
    parameters1 = 'matchrad=1 trirad=0.002 nobj=15 scale=1'
    # if file_path not in cache:
    #     # If not in the cache, read the ASCII file and store its content in the cache
    #     with open(file_path, 'r', encoding='ascii', errors='ignore') as file:
    #         file_content = file.read()
    #         cache[file_path] = file_content
            
    # else:
    #     # If in the cache, retrieve the content from the cache
        
    #     file_content = cache[file_path]
    #     print(type(file_content))
        
    
            
            
         
    command = f'match {file_path} 0 1 2 {path_catalog3} 0 1 2 {parameters1}'
        
                                            
    results = subprocess.run(command, shell=True, capture_output=True, text=True)
        
    stdout_str = results.stdout
    return stdout_str

pool = multiprocessing.Pool(2)
results = pool.map(call_match, ra_dec)

RA_center = []
DEC_center = []
sig = []
Nr = []

# Loop through results
for i, item in enumerate(results):
    RA1, DEC1 = ra_dec[i]
    
    # Extract 'sig' and 'Nr' values from 'item'
    match1_aux1 = item.find('sig=')
    match1_aux2 = item.find('Nr=')
    match1_auxsig1 = item[match1_aux1+4:match1_aux1+25]
    match1_auxnr1 = item[match1_aux2+3:match1_aux2+10]
    match1_sig1 = match1_auxsig1.split(' ', 1)[0]
    match1_nr1 = match1_auxnr1.split(' ', 1)[0]
    
    # Check if both 'sig' and 'Nr' values are non-empty
    if match1_sig1 and match1_nr1:
        RA_center.append(RA1)
        DEC_center.append(DEC1)
        sig.append(float(match1_sig1))  # Convert to float
        Nr.append(int(match1_nr1))  # Convert to int

# Create a structured NumPy array
match1_table1 = np.zeros(len(RA_center), dtype=[('RA_center', float), ('DEC_center', float), ('sig', float), ('Nr', int)])

# Fill the array with data
match1_table1['RA_center'] = RA_center
match1_table1['DEC_center'] = DEC_center
match1_table1['sig'] = sig
match1_table1['Nr'] = Nr


match1_table1 = np.sort(match1_table1, order='Nr')

if len(match1_table1) >= 3:
        a = match1_table1[0]['sig']
        b = match1_table1[1]['sig']
        c = match1_table1[2]['sig']
        if a <= b:
            if a <= c:
                i = 0
            else:
                i = 2
        else:
            if b <= c:
                i = 1
            else:
                i = 2

match1_RA = int(match1_table1[i][0])
match1_DEC = int(match1_table1[i][1])
path_catalog1 = '/home/alam/task/task-inspecity/notebooks/Catalog/Projected' +  '/cat_RA_'
path_catalog6 = str(match1_RA) + '_DEC_' + str(match1_DEC)
path_catalog7 = path_catalog1 + path_catalog6
parameters1 = 'matchrad=1 trirad=0.002 nobj=15 scale=1'
Match2 = 'match ' + '/home/alam/task/sext_t.csv' + ' 0 1 2 ' + path_catalog7 + ' 0 1 2 ' + parameters1
results = subprocess.run(Match2, shell=True, capture_output=True, text=True)

stdout_str = results.stdout

match1_aux5 = stdout_str.find('a=')
match1_aux6 = stdout_str.find('b=')
match1_aux7 = stdout_str.find('c=')
match1_aux8 = stdout_str.find('d=')
match1_aux9 = stdout_str.find('e=')
match1_aux10 = stdout_str.find('f=')
match1_aux11 = stdout_str.find('sig=')
match1_aux12 = stdout_str.find('Nr=')
match1_aux13 = stdout_str.find('Nm=')
match1_auxa1 = stdout_str[match1_aux5+2:match1_aux5+25]
match1_auxb1 = stdout_str[match1_aux6+2:match1_aux6+25]
match1_auxc1 = stdout_str[match1_aux7+2:match1_aux7+25]
match1_auxd1 = stdout_str[match1_aux8+2:match1_aux8+25]
match1_auxe1 = stdout_str[match1_aux9+2:match1_aux9+25]
match1_auxf1 = stdout_str[match1_aux10+2:match1_aux10+25]
match1_auxsig3 = stdout_str[match1_aux11+4:match1_aux11+25]
match1_auxnr3 = stdout_str[match1_aux12+3:match1_aux12+10]
match1_auxnm3 = stdout_str[match1_aux13+3:match1_aux13+10]
match1_sig3 = match1_auxsig3.split(' ', 1)[0]
match1_nr3 = match1_auxnr3.split(' ', 1)[0]
match1_nm3 = match1_auxnm3.split(' ', 1)[0]
match1_auxa2 = match1_auxa1.split(' ', 1)[0]
match1_auxb2 = match1_auxb1.split(' ', 1)[0]
match1_auxc2 = match1_auxc1.split(' ', 1)[0]
match1_auxd2 = match1_auxd1.split(' ', 1)[0]
match1_auxe2 = match1_auxe1.split(' ', 1)[0]
match1_auxf2 = match1_auxf1.split(' ', 1)[0]
match1_a = float(match1_auxa2)
match1_b = float(match1_auxb2)
match1_c = float(match1_auxc2)
match1_d = float(match1_auxd2)
match1_e = float(match1_auxe2)
match1_f = float(match1_auxf2)


match1_T = np.array([(match1_a), (match1_d)])
match1_R = np.array([(match1_b, match1_c), (match1_e, match1_f)])

match1_x_pix = 0
match1_y_pix = 0
match1_X_pix = np.array([(match1_x_pix), (match1_y_pix)])
match1_X_cielo = match1_T + np.dot(match1_R, match1_X_pix)
match1_RA_new = match1_X_cielo[0]
match1_DEC_new = match1_X_cielo[1]

f = 3.04 #mm
dep1_xi = match1_RA_new/f
dep1_eta = match1_DEC_new/f
dep1_RA_r = match1_RA*(np.pi/180)
dep1_DEC_r = match1_DEC*(np.pi/180)
dep1_arg1 = np.cos(dep1_DEC_r) - dep1_eta*np.sin(dep1_DEC_r)
dep1_arg2 = np.arctan(dep1_xi/dep1_arg1)
dep1_alpha1 = match1_RA + (180/np.pi)*dep1_arg2
dep1_arg3 = np.sin(dep1_arg2)
dep1_arg4 = dep1_eta*np.cos(dep1_DEC_r) + np.sin(dep1_DEC_r)
dep1_delta1 = (180/np.pi)*np.arctan((dep1_arg3*dep1_arg4)/dep1_xi)

print(f"RA1 = {dep1_alpha1}")
print(f"DEC1={dep1_delta1}")

cat_nor = '/home/alam/task/task-inspecity/notebooks/Catalog/Normal/'
new_cat1 = cat_nor + 'cat_RA_' + str(match1_RA) + '_DEC_' + str(match1_DEC)
with open(new_cat1, 'r') as file:
    lines = file.readlines()
data_new_cat1 = []
for line in lines:
    # Assuming data is whitespace-separated, modify accordingly if different
    elements = line.strip().split()  
    data_new_cat1.append(elements)

data_matched_b1 = []
data_matched_b2 = []
with open('/home/alam/task/task-inspecity/notebooks/matched.mtB', 'r') as file:
    lines = file.readlines()

for line in lines:
    # Assuming data is whitespace-separated, modify accordingly if different
    elements = line.strip().split()  
    data_matched_b1.append(elements)
with open('/home/alam/task/task-inspecity/notebooks/matched.mtB', 'r') as file:
    lines = file.readlines()

for line in lines:
    # Assuming data is whitespace-separated, modify accordingly if different
    elements = line.strip().split()  
    data_matched_b2.append(elements)


np_aux1 = int(data_matched_b1[0][0])
np_aux2 = int(data_matched_b2[0][0])

if np_aux1>np_aux2:
    np_cont1 = np_aux2
else:
    np_cont1 = np_aux1

rows = []

# Populate the list with rows
for i in range(len(data_matched_b1)):
    np_cont2 = int(data_matched_b1[i][0]) - np_cont1
    rows.append((data_new_cat1[np_cont2][0], data_new_cat1[np_cont2][1], data_new_cat1[np_cont2][2]))

np_table1 = np.array(rows)


dtype = [('xi_mm', float), ('eta_mm', float), ('mag', float)]

# Create an empty list to store rows
rows = []
conv1_largo1 = len(np_table1)

for index in range (0, conv1_largo1):
    conv1_alpha_d, conv1_delta_d, conv1_mag = np_table1[index]
    # conv1_alpha_d = np_table1[index][0]
    # conv1_delta_d = np_table1[index][1]
    # conv1_mag = np_table1[index][2]
    conv1_alpha_d = float(conv1_alpha_d)
    conv1_delta_d = float(conv1_delta_d)
    conv1_mag = float(conv1_mag)
    
    conv1_alpha_r = (np.pi/180)*conv1_alpha_d
    conv1_delta_r = (np.pi/180)*conv1_delta_d
    conv1_alpha_0_r = (np.pi/180)*dep1_alpha1
    conv1_delta_0_r = (np.pi/180)*dep1_delta1
    
    conv1_xi_up = np.cos(conv1_delta_r)*np.sin(conv1_alpha_r - conv1_alpha_0_r)
    conv1_xi_down = np.sin(conv1_delta_0_r)*np.sin(conv1_delta_r) + np.cos(conv1_delta_0_r)*np.cos(conv1_delta_r)*np.cos(conv1_alpha_r - conv1_alpha_0_r)
    conv1_xi = conv1_xi_up/conv1_xi_down
    
    conv1_eta_up = np.cos(conv1_delta_0_r)*np.sin(conv1_delta_r) - np.sin(conv1_delta_0_r)*np.cos(conv1_delta_r)*np.cos(conv1_alpha_r - conv1_alpha_0_r)
    conv1_eta_down = conv1_xi_down
    conv1_eta = conv1_eta_up/conv1_eta_down
    
    conv1_xi_mm = f*conv1_xi
    conv1_eta_mm = f*conv1_eta
    
    rows.append((conv1_xi_mm, conv1_eta_mm, conv1_mag))
cat_tran1 = np.array(rows, dtype=dtype)
output_path = 'new_cat.csv'

with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    
    # Write each row of the table to the CSV file
    for row in cat_tran1:
        writer.writerow(row)




new_parameters1 = 'trirad=0.002 nobj=20 max_iter=3 matchrad=1 scale=1'
Match4 = 'match ' + '/home/alam/task/new_cat.csv' + ' 0 1 2 ' + '/home/alam/task/task-inspecity/notebooks/new_cat' + ' 0 1 2 ' + new_parameters1
result_2 = subprocess.run(Match4, shell=True, capture_output=True, text=True)
stdout_str = result_2.stdout

match2_aux1 = stdout_str.find('a=')
match2_aux2 = stdout_str.find('b=')
match2_aux3 = stdout_str.find('c=')
match2_aux4 = stdout_str.find('d=')
match2_aux5 = stdout_str.find('e=')
match2_aux6 = stdout_str.find('f=')
match2_aux7 = stdout_str.find('sig=')
match2_aux8 = stdout_str.find('Nr=')
match2_auxa1 = stdout_str[match2_aux1+2:match2_aux1+25]
match2_auxb1 = stdout_str[match2_aux2+2:match2_aux2+25]
match2_auxc1 = stdout_str[match2_aux3+2:match2_aux3+25]
match2_auxd1 = stdout_str[match2_aux4+2:match2_aux4+25]
match2_auxe1 = stdout_str[match2_aux5+2:match2_aux5+25]
match2_auxf1 = stdout_str[match2_aux6+2:match2_aux6+25]
match2_auxsig4 = stdout_str[match2_aux7+4:match2_aux7+25]
match2_auxnr4 = stdout_str[match2_aux8+3:match2_aux8+10]
match2_auxa2 = match2_auxa1.split(' ', 1)[0]
match2_auxb2 = match2_auxb1.split(' ', 1)[0]
match2_auxc2 = match2_auxc1.split(' ', 1)[0]
match2_auxd2 = match2_auxd1.split(' ', 1)[0]
match2_auxe2 = match2_auxe1.split(' ', 1)[0]
match2_auxf2 = match2_auxf1.split(' ', 1)[0]
match2_sig = match2_auxsig4.split(' ', 1)[0]
match2_nr = match2_auxnr4.split(' ', 1)[0]
match2_a = float(match2_auxa2)
match2_b = float(match2_auxb2)
match2_c = float(match2_auxc2)
match2_d = float(match2_auxd2)
match2_e = float(match2_auxe2)
match2_f = float(match2_auxf2)

match2_T = np.array([(match2_a), (match2_d)])
match2_R = np.array([(match2_b, match2_c), (match2_e, match2_f)]) 

match2_x_pix = 0
match2_y_pix = 0
match2_X_pix = np.array([(match2_x_pix), (match2_y_pix)])
match2_X_cielo = match2_T + np.dot(match2_R, match2_X_pix)
match2_RA_new = match2_X_cielo[0]
match2_DEC_new = match2_X_cielo[1]

match2_roll_r = np.arctan2(match2_c, match2_b)
match2_roll_d = (180/np.pi)*match2_roll_r


dep2_xi = match2_RA_new/f
dep2_eta = match2_DEC_new/f
dep2_RA_r = dep1_alpha1*(np.pi/180)
dep2_DEC_r = dep1_delta1*(np.pi/180)
dep2_arg1 = np.cos(dep2_DEC_r) - dep2_eta*np.sin(dep2_DEC_r)
dep2_arg2 = np.arctan(dep2_xi/dep2_arg1)
dep2_alpha1 = dep1_alpha1 + (180/np.pi)*dep2_arg2
dep2_arg3 = np.sin(dep2_arg2)
dep2_arg4 = dep2_eta*np.cos(dep2_DEC_r) + np.sin(dep2_DEC_r)
dep2_delta1 = (180/np.pi)*np.arctan((dep2_arg3*dep2_arg4)/dep2_xi)


print(f"RA2 = {dep2_alpha1}")
print(f"DEC2 = {dep2_delta1}")

time5 = time.time()

print ('Processing time:')
print ('- Imports:       ', time2 - time1, 'seconds.')
print ('- Picture:       ', time3 - time2, 'seconds.')
print ('- SExtractor:    ', time4 - time3, 'seconds.')
print ('- Match routines:', time5 - time4, 'seconds.')
print ('- Total time:    ', time5 - time1, 'seconds.')

