import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


def get_data_from_keys(data, indep_key, dep_key):
    global result_dict
    dep = np.array(data[dep_key])
    indep = np.array(data[indep_key])
    not_nan_filter = ~(pd.isna(dep) | pd.isna(indep))
    all_samples = len(dep)
    dep = dep[not_nan_filter]
    indep = indep[not_nan_filter]
    found_samples = len(dep)
    try:
        float(indep[0])
    except:
        get_num = lambda x: float(x[0])
        indep = np.array([get_num(i) for i in indep])
    indep_train, indep_test, dep_train, dep_test = train_test_split(indep, dep, train_size=0.8)
    return indep, indep_train, indep_test, dep, dep_train, dep_test, found_samples/all_samples


def extract_selected_keys(selected_keys, data):
    arrs = []
    y = np.array(data['mrs'])
    dep_nan_filter = pd.isna(y)
    for k in selected_keys:
        arr = np.array(data[k])
        arr = np.expand_dims(arr, 1)
        arrs.append(arr)
    x = np.concatenate(arrs, axis=1)
    y = np.array(data['mrs'])
    return x, y



def get_performance(indep_train, dep_train, indep_test, dep_test, mode):
    global result_dict
    model = sm.OLS(dep_train, indep_train)
    results = model.fit()
    predictions = model.predict(params=results.params, exog=np.expand_dims(indep_test, 1))
    predictions[predictions > 6] = 6
    predictions[predictions < 0] = 0
    predictions = predictions.round()
    result_dict['p-value' + mode].append(results.pvalues[0])
    result_dict['accuracy' + mode].append((predictions == dep_test).mean())
    result_dict['accuracy_+-1' + mode].append(np.mean(np.abs(predictions-dep_test)<=1))
    result_dict['accuracy_+-2' + mode].append(np.mean(np.abs(predictions-dep_test)<=2))
    result_dict['bigger_than_2' + mode].append(np.mean((predictions>2)==(dep_test>2)))


path_csv = r'C:\Users\Fabian\Desktop\sample_lesion\tabular_data.csv'
path_output = r'C:\Users\Fabian\Desktop\sample_lesion\linear_feature_analysis.csv'
result_dict = {'Feature': [], 'Feature availability': [], "p-value_all_data": [], 'accuracy_all_data': [],
               'accuracy_+-1_all_data': [], 'accuracy_+-2_all_data': [], 'bigger_than_2_all_data': [], 'p-value_train_test_split': [], 'accuracy_train_test_split': [],
               'accuracy_+-1_train_test_split': [], 'accuracy_+-2_train_test_split': [], 'bigger_than_2_train_test_split': []}

data = pd.read_csv(path_csv)
keys = np.array(data.keys())
dep_key = keys[3]
indep_keys = np.concatenate((keys[2:3], keys[4:]))


# select keys with more than 20% of data available & +-1 accuracy of more than 79%
selected_keys = ['age', 'hypertension', 'treatment', 'baselinenihssscore', 'norm_nihss', 'norm_age', 'didthepatienthaveahemorrhage',
                 'd90nihss', 'd1nihss', 'lesion volume']

gbdt_keys = ['lesion volume', 'age', 'baselinenihssscore', 'priorstroketia', 'treatment', 'tpa', 'male', 'myocardialinfarction',
             'hypertension', 'atrialfibrillation', 'hypercholesterolemia', 'diabetes', 'haspatienteverhadastrokepriortoq',
             'onset_to_ER', 'pre-eventrankin', '1a. Level of Consciousness', '1b. LOC Questions', '1c. LOC Commands ',
             '2. Best Gaze', '3. Visual Fields', '4. Facial Palsy', '5a. Motor: Left Arm', '5b. Motor: Right Arm', '6a. Motor: Left leg',
             '6b. Motor: Right Leg', '7. Limb Ataxia', '8. Sensory', '9. Best Language', '10. Dysarthria', '11. Extinction and Inattention',
             'Initial Systolic Blood Pressure on arrival at STUDY SITE', 'Initial Diastolic Blood Pressure on arrival at STUDY SITE',
             'Glucose']
x, y = extract_selected_keys(selected_keys, data)

# indep_train, indep_test, dep_train, dep_test = train_test_split(indep, dep, train_size=0.8)
for indep_key in indep_keys:
    print(indep_key)
    indep, indep_train, indep_test, dep, dep_train, dep_test, availability = get_data_from_keys(data, indep_key, dep_key)
    if len(dep) < 10:
        continue
    result_dict['Feature'].append(indep_key)
    result_dict['Feature availability'].append(f'{len(dep)} out of 491')
    get_performance(indep, dep, indep, dep, '_all_data')
    get_performance(indep_train, dep_train, indep_test, dep_test, '_train_test_split')

result = pd.DataFrame.from_dict(result_dict)
result.to_csv(path_output)
print('well done!')