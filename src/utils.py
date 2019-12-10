import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import re
import geopandas as gp

county_borough_map = {'Kings':'B', 
 'New York':'M', 
 'Richmond':'R',
'Queens':'Q',
'Bronx':'X'}


class CleanUtils:
	'''
	Various cleaning and transforming functions used throughout the project
	'''
	def __init__(self):
		pass

	def _dummy(self, df, column):
		return pd.concat((df, pd.get_dummies(df[column], prefix=column, drop_first=True)), axis=1).drop(column, axis=1)

	def find_zip_from_address(self, df):
		'''Pulls zipcodes from the Playground Location column and fills in missing zipcodes'''
		df = df.copy() # prevent mutation to original dataframe
		# Searches Playground_Location series for presence of a zipcode on rows where zip is missing
		zipcodes = df[(df.Zip.isna())]['Playground_Location'].str.extract(r'(\d{5})')
		df.loc[(df.Zip.isna()),'Zip'] = zipcodes.values
		return df

	def coord_to_zip(self, df, geolocator):
		'''Converts coordinates to ZIP using pandas .apply() and geopy'''

		df = df.copy()
		df2 = df.replace(np.NaN,-1) # .apply() fails with Null values, this is to bypass that
		df['Zip'] = df2.apply(self._coord_to_zip, axis=1,args=(geolocator,))
		return df

	def _coord_to_zip(self, row, geolocator):
		'''Pandas .apply() function to search for a zipcode from coordinates'''
		if row['Zip'] != -1:#If zip is present, return zip as is
			return row['Zip']
		# If Zip is missing:
		if row['lat'] != -1.0: #If coordinates are present
			coordinate_query = '{}, {}'.format(row['lat'], row['lon'])
			address = geolocator.reverse(coordinate_query).raw['address']
			if 'postcode' in address:
				return address['postcode'][:5]
		else:
			return np.NaN


	def clean_multiple_zip_playgrounds(self, df, geolocator):
		'''
		Some playgrounds in the dataset contain lists of zipcodes, instead
		of a single zip code (after reviewing it, it appears to be a data entry error).
		This code attempts to find the correct zipcode to use. If no zipcode is corrext, 
		'''

		df = df.copy()
		df.dropna(axis=0, subset=['Zip'], inplace=True)
		df.reset_index(drop=True, inplace=True)
		df.Zip = df.Zip.str.split(',')

		df.Zip = df.Zip.apply(self._clean_bad_zips)
		df.Zip = df.Zip.apply(set).apply(list)
		
		# Searches a zipcode from coordinates for all entries with multiple zips
		df.loc[(df.Zip.apply(len)>1), 'zipcode_geopy'] = \
			df.loc[(df.Zip.apply(len)>1)].apply(
				self._coord_to_zip_multizips, args=(geolocator,), axis=1)

		df.loc[(df.Zip.apply(len)>1), 'Zip'] = \
		df.loc[(df.Zip.apply(len)>1)].apply(
			self._zip_match, axis=1)

		df.Zip = df.Zip.apply(lambda x: x[0] if len(x) == 1 else x)
		return df


	def _zip_match(self, row):
		'''Returns a Zipcode from zipcode_geopy column if it matches 
		a zip in the list of zipcodes for a given entry'''
		A, B = set(row['Zip']), set([row['zipcode_geopy']])
		if len(A.intersection(B)) == 1:
			return row['zipcode_geopy']
		else:
			return row['Zip'][0]

	def _clean_bad_zips(self, row):
		'''Filters out any zipcodes that are less than 5 digits from a list'''
		return [zipcode.strip() for zipcode in row if len(zipcode.strip()) >= 5]


	def _coord_to_zip_multizips(self, row, geolocator):
		coordinate_query = '{}, {}'.format(row['lat'], row['lon'])
		if geolocator.reverse(coordinate_query).raw:
			try:
				zipcode = geolocator.reverse(coordinate_query).raw['address']['postcode'][:5]
				# additional check to ensure correct zipcode
				if len(zipcode) == 5 and int(zipcode) < 20000:
					return zipcode
				else:
					return np.NaN
			except:
				return np.NaN
		else:
			return np.NaN	


class CleanParks(CleanUtils):
	'''
	Functions to transform data from it's various sources, into a useable one
	'''
	def __init__(self):
		pass

	def nyc_playgrounds_dataframe(self, df):
		df = df.copy()
		# Fill in missing Zipcodes with Park_Location column
		df = self.find_zip_from_address(df)
		# Fill in missing remaining missing zipcodes with Geopy
		geolocator = Nominatim(user_agent="specify_your_app_name_here")
		df = self.coord_to_zip(df, geolocator)
		# Drop remaining null zipcodes
		df = df[df.Zip.notnull()].reset_index(drop=True)
		# Clean entries that contain multiple zipcodes in one row
		df = self.clean_multiple_zip_playgrounds(df, geolocator)
		df = self.clean_accessible_adaptive_cols(df)
		df = self.clean_school(df)
		df = self.clean_status(df)
		df = self.clean_level(df)
		df = self.aggregate_nyc_playgrounds(df)
		return df

	def tax_returns_dataframe(self, tax_returns):
		'''Cleans the tax_returns dataframe'''
		tax_returns_ny = tax_returns.loc[tax_returns['STATE']=='NY',:].reset_index(drop=True)
		tax_returns_ny = tax_returns_ny.groupby('zipcode').sum()
		tax_returns_ny.drop(['STATEFIPS', 'agi_stub'], axis=1, inplace=True)
		return tax_returns_ny

	def zipcode_dataframe(self, df):
		df = df.copy()
		df['Borough'] = df['COUNTY'].map(county_borough_map)
		zip_df = df[['ZIPCODE', 'Borough', 'POPULATION', 'AREA', 'geometry']]
		zip_df.drop_duplicates(subset='ZIPCODE', keep='first', inplace=True)
		zip_df.reset_index(drop=True, inplace=True)
		zip_df.ZIPCODE= zip_df.ZIPCODE.astype('int64')
		return zip_df

	def clean_accessible_adaptive_cols(self, df):
		df = df.copy()
		df['Adaptive_Swing'] = df['Adaptive_Swing'].replace({'':np.NaN})
		for col in ['Accessible', 'Adaptive_Swing']:
			df[col] = df[col].replace({'N':0, 'Y':1})
			df[col].fillna(0, inplace=True)
		return df

	def clean_school(self, df):
		df = df.copy()
		df['School_ID'].fillna(0,inplace=True)
		df['School'] = np.where(df['School_ID'] != 0, 1, 0)
		df.drop('School_ID', inplace=True, axis=1)
		return df

	def clean_status(self, df):
		df = df.copy()
		df.Status = df.Status.str.lower()
		df.Status.fillna('None', inplace=True)
		df.Status.loc[df['Status'].str.contains('closed')] = 'closed'
		df.Status.loc[df['Status'].str.contains('weekend')] = 'weekends'
		df.Status.loc[df['Status'].str.contains('weekday')] = 'weekdays'
		df.Status.loc[df['Status'].str.contains('two playgr')] = 'two playgrounds'
		df.Status.loc[df['Status'].str.contains('open to the public')] = 'open to the public'
		# Dummy column
		df = self._dummy(df, 'Status')
		return df


	def clean_level(self, df):
		'''Fills null values in clean_level column. If accessible = 0, fillna(0), else fillna(1)
		Filling null values with level 1 will add some bias'''
		df = df.copy()
		df.Level.loc[df['Accessible'] == 0.0] = df.Level[df['Accessible'] == 0.0].fillna(0)
		df['Level'].fillna(1, inplace=True)
		df = self._dummy(df, 'Level')
		return df

	def aggregate_nyc_playgrounds(self, df):
		'''Dropping object type columns, and other columns that can't be simply aggregated. 
		Also doing a groupby operation for each zipcode.'''
		df = df.copy()
		df.Zip = df.Zip.astype('int64')
		df.drop(df.select_dtypes(include=['object']).columns.tolist(), axis=1,
	       inplace=True)
		df.drop(['lat', 'lon'], axis=1, inplace=True)
		df['playground_count'] = [1]*len(df)
		df = df.groupby('Zip').sum().reset_index()
		return df


def main():
	parks = pd.read_json('./data/parks/DPR_Parks_001.json')
	playgrounds = pd.read_json('./data/playgrounds/DPR_Playgrounds_001.json')
	tax_returns = pd.read_csv('./data/income/16zpallagi.csv')
	zipcode_shape_data = gp.read_file('./data/ZIP_CODE_040114 (3)/ZIP_CODE_040114.shp').to_crs(epsg=4326) # to_crs converts the coordinate system to familiar form
	
	# Clean data
	clean = CleanParks()
	CU = CleanUtils()
	tax_returns_ny = clean.tax_returns_dataframe(tax_returns)
	zipcode_df = clean.zipcode_dataframe(zipcode_shape_data)
	
	# Merge zipcode data with tax returns data
	tax_returns_df = pd.merge(zipcode_df, tax_returns_ny.reset_index(), how='inner', 
		left_on='ZIPCODE', right_on='zipcode').set_index('ZIPCODE')

	# Merge Parks and playgrounds
	playgrounds.rename(columns={'Name':'Playground_Name', 'Location':'Playground_Location'}, inplace=True)
	parks.rename(columns={'Name':'Park_Name', 'Location':'Park_Location'}, inplace=True)
	nyc_playgrounds = pd.merge(playgrounds, parks, how='left', on='Prop_ID')

	# Clean merged dataframe
	nyc_playgrounds = clean.nyc_playgrounds_dataframe(nyc_playgrounds)

	# Merge nyc_playgrounds and Tax Returns  
	df = pd.merge(nyc_playgrounds, tax_returns_df, how='right', left_on='Zip', 
		right_on='ZIPCODE').set_index('zipcode').drop('Zip', axis=1)
	df['borough_name'] = df['Borough']
	df = CU._dummy(df, 'Borough')

	return df



if __name__ == "__main__":
	df = main()
	df.to_csv('./data/nyc_playgrounds.csv', index=None)

