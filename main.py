from src.preprocessing import all_data_filtered
from src.features import images_features_cache
from src.models import LSTM, GRU
from src.cross_validation import  run_cv, naive_baseline



labels = [1 if e['label'] == 'ASD' else 0 for e in all_data_filtered]
groups = [e['participant_id'] for e in all_data_filtered]



results = {}

results['naive baseline'] = naive_baseline()
results['baseline'] = run_cv(all_data_filtered, labels, groups, images_features_cache, 'baseline', model_class=LSTM, use_duration=True, shuffle_fixations=False)
results['shuffled order'] = run_cv(all_data_filtered, labels, groups, images_features_cache, 'shuffled order', model_class=LSTM, use_duration=True, shuffle_fixations=True)
results['no duration'] = run_cv(all_data_filtered, labels, groups, images_features_cache, 'no duration', model_class=LSTM, use_duration=False, shuffle_fixations=False)
results['GRU'] = run_cv(all_data_filtered, labels, groups, images_features_cache, 'GRU', model_class=GRU, use_duration=True, shuffle_fixations=False)