import time
time1 = time.time()
import os
import subprocess
from astropy.io import fits,ascii
from astropy.table import Table
import multiprocessing
from astropy.table import Table
import numpy as np

time2 = time.time()


time3 = time.time()


sext = 'source-extractor ' + '/home/alam/task/task-inspecity/source-ex/img_fits.fits'

subprocess.run(sext, shell=True,capture_output=True, text=True)
sex_aux1 = ascii.read('/home/alam/task/task-inspecity/source-ex/test.cat', format='sextractor')

sex_aux1.sort(['MAG_ISO'])
sex_aux2 = sex_aux1[0:40]

sex_x = sex_aux2['X_IMAGE']
sex_y = sex_aux2['Y_IMAGE']
sex_mag = sex_aux2['MAG_ISO']

sex_x1 = (sex_x - 512)*0.00270   
sex_y1 = (sex_y - 512)*0.00270  

ascii.write([sex_x1, sex_y1, sex_mag], 'sext_t', delimiter = ' ', format = 'no_header',overwrite=True, formats = {'col0':'% 15.10f', 'col1':'% 15.10f', 'col2':'% 15.10f'})

time4 = time.time()

parameters1 = 'trirad=0.002 nobj=15 max_iter=1 matchrad=1 scale=1'

match1_tabla1 = Table(names=('RA_center', 'DEC_center', 'sig', 'Nr'))


ra_dec = [(ra, dec) for ra in range(0, 360, 25) for dec in range(-60, 90, 25)]

def call_match(ra_dec):
    RA1, DEC1 = ra_dec
    # Transform RA and DEC in string and make the path for catalog.
    path_catalog1 = '/home/alam/task/task-inspecity/notebooks/Catalog/Projected' +  '/cat_RA_'
    path_catalog2 = str(RA1) + '_DEC_' + str(DEC1)
    path_catalog3 = path_catalog1 + path_catalog2
    # Do Match.
    parameters1 = 'matchrad=1 trirad=0.002 nobj=15 scale=1'
    command = 'match ' + '/home/alam/task/task-inspecity/sex_t' + ' 0 1 2 ' + path_catalog3 + ' 0 1 2 ' + parameters1
    
    results = subprocess.run(command, shell=True, capture_output=True, text=True)
    stdout_str = results.stdout
    return stdout_str

pool = multiprocessing.Pool(2)
results = pool.map(call_match, ra_dec)

match1_table1 = Table(names=('RA_center', 'DEC_center', 'sig', 'Nr'))

for i, item in enumerate(results):
    RA1, DEC1 = ra_dec[i]
    
    
    match1_aux1 = item.find('sig=')
    match1_aux2 = item.find('Nr=')
    match1_auxsig1 = item[match1_aux1+4:match1_aux1+25]
    match1_auxnr1 = item[match1_aux2+3:match1_aux2+10]
    match1_sig1 = match1_auxsig1.split(' ', 1)[0]
    match1_nr1 = match1_auxnr1.split(' ', 1)[0]
    if match1_sig1 and match1_nr1:  # Check if both 'sig' and 'Nr' values are non-empty
            match1_table1.add_row([str(RA1), str(DEC1), match1_sig1, match1_nr1])

match1_table1.sort('Nr',reverse=True)

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
print(match1_table1)

match1_RA = int(match1_table1[i][0])
match1_DEC = int(match1_table1[i][1])
path_catalog1 = '/home/alam/task/task-inspecity/notebooks/Catalog/Projected' +  '/cat_RA_'
path_catalog6 = str(match1_RA) + '_DEC_' + str(match1_DEC)
path_catalog7 = path_catalog1 + path_catalog6
parameters1 = 'matchrad=1 trirad=0.002 nobj=15 scale=1'
Match2 = 'match ' + '/home/alam/task/task-inspecity/sex_t' + ' 0 1 2 ' + path_catalog7 + ' 0 1 2 ' + parameters1
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
new_cat2 = ascii.read(new_cat1)
np_matched_B1 = ascii.read('/home/alam/task/task-inspecity/notebooks/matched.mtB')
np_matched_B2 = ascii.read('/home/alam/task/task-inspecity/notebooks/matched.unB')
np_aux1 = np_matched_B1[0][0]
np_aux2 = np_matched_B2[0][0]

if np_aux1>np_aux2:
    np_cont1 = np_aux2
else:
    np_cont1 = np_aux1

np_table1 = Table([[], [], []])

for i in range(0, len(np_matched_B1), 1):
    np_cont2 = np_matched_B1[i][0] - np_cont1
    np_table1.add_row([new_cat2[np_cont2][0], new_cat2[np_cont2][1], new_cat2[np_cont2][2]])




cat_tran1 = Table([[], [], []])
conv1_largo1 = len(np_table1)

for index in range (0, conv1_largo1):
    conv1_alpha_d = np_table1[index][0]
    conv1_delta_d = np_table1[index][1]
    conv1_mag = np_table1[index][2]
    
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
    
    cat_tran1.add_row([conv1_xi_mm, conv1_eta_mm, conv1_mag])


ascii.write(cat_tran1, 'new_cat', delimiter = ' ', format = 'no_header', overwrite=True,formats = {'col0':'% 15.5f', 'col1':'% 15.5f', 'col2':'% 15.2f'})

new_parameters1 = 'trirad=0.002 nobj=20 max_iter=3 matchrad=1 scale=1'
Match4 = 'match ' + '/home/alam/task/task-inspecity/sex_t' + ' 0 1 2 ' + '/home/alam/task/task-inspecity/notebooks/new_cat' + ' 0 1 2 ' + new_parameters1
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

