from os.path import dirname, abspath, join

root_path = dirname(dirname(dirname(abspath(__file__))))
data_path = join(root_path, 'data')
cleaned_data_path = join(data_path, 'cleaned_data')
# %%
