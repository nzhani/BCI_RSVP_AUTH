import json
import mne
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# 1. Load data to evaluate dynamically
raw = mne.io.read_raw_brainvision("c:/Users/nzhan/Downloads/BCI assignment/RSVP_login_VPeal.vhdr", preload=True)
raw.rename_channels({'EOGv1': 'Fp1'})
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage, on_missing="warn")
raw.filter(l_freq=None, h_freq=40., fir_design='firwin', verbose=False)
raw.resample(sfreq=100, npad='auto', window='boxcar', verbose=False)

events, event_id_from_annotations = mne.events_from_annotations(raw, verbose=False)

event_id = {'T1': 1, 'T2': 2, 'T3': 3,
            'NT1':4, 'NT2': 5, 'NT3': 6, 'NT4':7, 'NT5':8, 'NT6':9, 'NT7':10,
            'NT8':11, 'NT9':12, 'NT10':13, 'NT11':14, 'NT12':15, 'NT13':16,
            'NT14':17, 'NT15':18, 'NT16':19, 'NT17':20, 'NT18':21, 'NT19':22,
            'NT20':23, 'NT21':24, 'NT22':25}

epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1, baseline=(None, 0), detrend=1, preload=True, verbose=False)

epochs = mne.epochs.combine_event_ids(epochs, ['T1','T2','T3'], {'T': 112}, copy=True)
epochs = mne.epochs.combine_event_ids(epochs, ['NT1', 'NT2', 'NT3', 'NT4', 'NT5', 'NT6', 'NT7', 'NT8', 'NT9',
                                               'NT10', 'NT11', 'NT12', 'NT13', 'NT14', 'NT15', 'NT16', 'NT17',
                                               'NT18', 'NT19', 'NT20', 'NT21', 'NT22'], {'NT': 113}, copy=True)

# Artifact Rejection
veog_epoch_ch = 'Fp1'
veog_ep = epochs.copy().pick([veog_epoch_ch]).get_data(copy=True)[:, 0, :]
veog_range = veog_ep.max(axis=1) - veog_ep.min(axis=1)
bad_epochs = veog_range > 150e-6
bad_numbers = np.flatnonzero(bad_epochs)
epochs_clean = epochs.copy()
epochs_clean.drop(bad_numbers, reason='ocular_artifact')

# Feature Extraction test
roi_candidates = ["Cz", "Pz", "CP1", "CP2", "P3", "P4", "CPz"]
roi = [ch for ch in roi_candidates if ch in epochs_clean.ch_names]

tmin_p300, tmax_p300 = 0.2, 0.6
times = epochs_clean.times
time_mask = (times >= tmin_p300) & (times <= tmax_p300)

X_raw = epochs_clean.copy().pick(roi).get_data(copy=True)[:, :, time_mask]
n_epochs, n_channels, n_times = X_raw.shape
X = X_raw.reshape(n_epochs, -1) * 1e6  # Flatten: (epochs, channels * times)

y = (epochs_clean.events[:, 2] == 112).astype(int)

# Classifier 1: LDA
clf1 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred1 = cross_val_predict(clf1, X, y, cv=cv, method='predict')
y_proba1 = cross_val_predict(clf1, X, y, cv=cv, method='predict_proba')[:, 1]
print(f"LDA -> AUC: {roc_auc_score(y, y_proba1):.4f}, BA: {balanced_accuracy_score(y, y_pred1):.4f}")

# Classifier 2: LogReg (Balanced)
clf2 = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs'))
y_pred2 = cross_val_predict(clf2, X, y, cv=cv, method='predict')
y_proba2 = cross_val_predict(clf2, X, y, cv=cv, method='predict_proba')[:, 1]
print(f"LogReg -> AUC: {roc_auc_score(y, y_proba2):.4f}, BA: {balanced_accuracy_score(y, y_pred2):.4f}")

# Feature approach 2: Time Window Averages on ROI
t_windows = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7)]
X_win = np.zeros((n_epochs, len(roi) * len(t_windows)))
epochs_data = epochs_clean.copy().pick(roi).get_data(copy=True) * 1e6
for i in range(n_epochs):
    idx = 0
    for ch_idx in range(len(roi)):
        for tmin, tmax in t_windows:
            mask = (times >= tmin) & (times <= tmax)
            X_win[i, idx] = epochs_data[i, ch_idx, mask].mean()
            idx += 1

y_pred3 = cross_val_predict(clf2, X_win, y, cv=cv, method='predict')
y_proba3 = cross_val_predict(clf2, X_win, y, cv=cv, method='predict_proba')[:, 1]
print(f"LogReg (Win Avg) -> AUC: {roc_auc_score(y, y_proba3):.4f}, BA: {balanced_accuracy_score(y, y_pred3):.4f}")

