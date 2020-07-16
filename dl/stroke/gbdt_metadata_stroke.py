from dl.stroke.stroke_gbdt import *

path_csv = r'C:\Users\Fabian\Desktop\sample_lesion\tabular_data2.csv'
path_output = r'C:\Users\Fabian\Desktop\sample_lesion'
np.random.seed(42)

data = pd.read_csv(path_csv)
keys = np.array(data.keys())
dep_key = keys[3]
indep_keys = np.concatenate((keys[2:3], keys[4:]))

stroke_keys = ['lesion volume', 'age', 'baselinenihssscore', 'priorstroketia', 'treatment', 'tpa', 'male', 'myocardialinfarction',
             'hypertension', 'atrialfibrillation', 'hypercholesterolemia', 'diabetes', 'haspatienteverhadastrokepriortoq',
             'pre-eventrankin', '1a. Level of Consciousness', '1b. LOC Questions', '1c. LOC Commands ',
             '2. Best Gaze', '3. Visual Fields', '4. Facial Palsy', '5a. Motor: Left Arm', '5b. Motor: Right Arm', '6a. Motor: Left leg',
             '6b. Motor: Right Leg', '7. Limb Ataxia', '8. Sensory', '9. Best Language', '10. Dysarthria', '11. Extinction and Inattention',
             'Initial Systolic Blood Pressure on arrival at STUDY SITE', 'Initial Diastolic Blood Pressure on arrival at STUDY SITE',
             'Glucose']


stroke_keys += ['tci_num', 'onset_to_imaging', 'image_to_dsa', 'image_to_eariest_treat']

x, y, train_test_split, x_names = get_stroke_xy(data, stroke_keys)
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'multiclass', # default is 'multiclass'
#     'num_classes': 3,
#     # 'objective': 'regression', # default is 'regression'
#     # 'metric': 'mse',
#     'max_leaves': 50, #default is 50
#     'max_depth': 5,  # 9 optimal for activations, 4 optimal for metadata only
#     'max_bin': 255, # default is 255
#     'num_iterations': 4000,
#     'learning_rate': 0.0005,  # 0.0006 optimal for metadata, 0.0045 for activations
#     'feature_fraction': 0.8, # 0.8 for metadata only
#     'bagging_fraction': 0.5,
#     'bagging_freq': 2,
#     'verbose': 0,
#     'min_data_in_leaf': 25 # default is 9
# }

params = {
    'boosting_type': 'gbdt',
    # 'objective': 'multiclass', # default is 'multiclass'
    # 'num_classes': 3,
    'objective': 'regression', # default is 'regression'
    'metric': 'mse',
    'max_leaves': 200, #default is 50
    'max_depth': 4,  # 9 optimal for activations, 4 optimal for metadata only
    'max_bin': 255, # default is 255
    'num_iterations': 4000,
    'learning_rate': 0.0015,  # 0.0006 optimal for metadata, 0.0045 for activations
    'feature_fraction': 0.8, # 0.8 for metadata only
    'bagging_fraction': 0.5,
    'bagging_freq': 2,
    'verbose': 0,
    'min_data_in_leaf': 33 # default is 9
}

cat_features = list(range(len(stroke_keys)))
out_path = r'C:\Users\Fabian\Desktop\sample_lesion\gbdt_metadata'
# params['categorical_feature'] = cat_features[3:]
cross_validation_gbdt(data, params, group_mrs=False, delta=True, stroke_keys=stroke_keys, output_path=out_path)
print('db')