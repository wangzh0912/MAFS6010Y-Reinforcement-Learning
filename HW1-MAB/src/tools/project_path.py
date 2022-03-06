from os.path import dirname, abspath, join

root_path = dirname(dirname(dirname(abspath(__file__))))
data_path = join(root_path, 'data')
price_volume_path = join(data_path, 'price_volume')
cleaned_data_path = join(data_path, 'cleaned_data')
# %%
