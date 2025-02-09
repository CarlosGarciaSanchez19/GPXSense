
import gpxpy as gpx
import geopandas as gpd
import contextily as ctx
import folium
import webbrowser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import random as rnd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class gpxAnalyseClass:

	units: dict = {
		'lat': None,
		'lon': None,
		'ele': "m",
		'time': None,
		'hr': "beats/min",
		'cad': "steps/min",
		'dist': "m",
		'time_acc': "min",
		'pace': "min/Km"
	}

	names: dict = {
		'lat': "Latitude",
		'lon': "Longitude",
		'ele': "Altitude",
		'time': "Time and date",
		'hr': "Heart rate",
		'cad': "Cadence",
		'dist': "Distance",
		'time_acc': "Time",
		'pace': "Pace"
	}

	def __init__(self, gpx_filename):
		with open(gpx_filename, "r") as gpx_file:
			gpx_object = gpx.parse(gpx_file)
		self.gpx_filename = gpx_filename.split('.')[0]
		self.df = self.gpx_to_df(gpx_object)
		self.df['cad'] = self.df['cad'] + 100
		self.df['ele'] = self.df['ele'] - self.df['ele'].min()
		self.df['dist'] = self.get_distances()
		self.df['time_acc'] = self.get_times()
		self.df['pace'] = self.get_speeds()
		for i in range(len(self.df)):
			if self.df['pace'][i] > 20:
				self.df.loc[i, 'pace'] = 20
		df_numeric = self.df.select_dtypes(include=['number'])
		self.df_norm = (df_numeric - df_numeric.mean()) / df_numeric.std()
		self.df_norm['pace'] = -self.df_norm['pace']

	def gpx_to_df(self, gpx_object):
		lat = []
		lon = []
		ele = []
		time = []
		hr = []
		cad = []
		for track in gpx_object.tracks:
			for segment in track.segments:
				for point in segment.points:
					lat.append(point.latitude)
					lon.append(point.longitude)
					ele.append(point.elevation)
					time.append(point.time)
					if point.extensions is not None:
						for ext in point.extensions:
							for child in ext:
								if child.tag.endswith("hr"):
									hr.append(float(child.text))
								elif child.tag.endswith("cad"):
									cad.append(float(child.text))
		if point.extensions is not None:
			return pd.DataFrame({"lat": lat, "lon": lon, "ele": ele, "time": time, "hr": hr, "cad": cad})
		else:
			return pd.DataFrame({"lat": lat, "lon": lon, "ele": ele, "time": time})

	def get_distance(self, lat1, lon1, lat2, lon2):
			R = 6371e3  # metres
			φ1 = np.radians(lat1)
			φ2 = np.radians(lat2)
			Δφ = np.radians(lat2 - lat1)
			Δλ = np.radians(lon2 - lon1)

			a = np.sin(Δφ / 2) * np.sin(Δφ / 2) + np.cos(φ1) * np.cos(φ2) * np.sin(Δλ / 2) * np.sin(Δλ / 2)
			c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

			return R * c

	def get_speed(self, lat1, lon1, lat2, lon2, time1, time2):
		distance = self.get_distance(lat1, lon1, lat2, lon2) / 1000
		time = (time2 - time1).total_seconds() / 60
		if distance == 0:
			return 20.0
		else:
			return time / distance

	def get_distances(self):
		distances = [0.0]
		accumulated_distance = 0.0
		for i in range(1, len(self.df)):
			accumulated_distance += self.get_distance(self.df.lat[i - 1], self.df.lon[i - 1], self.df.lat[i], self.df.lon[i])
			distances.append(accumulated_distance)
		return distances

	def get_speeds(self):
		speeds = [20.0]
		for i in range(1, len(self.df)):
			speeds.append(self.get_speed(self.df.lat[i - 1], self.df.lon[i - 1],
						self.df.lat[i], self.df.lon[i],
						self.df.time[i - 1], self.df.time[i]))
		return speeds

	def get_times(self):
		times = [0.0]
		accumulated_time = 0.0
		for i in range(1, len(self.df)):
			accumulated_time += (self.df['time'][i] - self.df['time'][i - 1]).total_seconds() / 60
			times.append(accumulated_time)
		return times


	def get_gdf(self, crs):
		return gpd.GeoDataFrame(self.df, geometry=gpd.points_from_xy(self.df.lon, self.df.lat), crs=crs)

	def plot_track_satelite(self, filename=None, cmap='viridis', parameter=None):
		gdf = self.get_gdf(crs="EPSG:4326")
	
		x_max = gdf.geometry.x.max()
		y_max = gdf.geometry.y.max()
		scale = 5
		if x_max < y_max:
			fig_size_x = scale
			fig_size_y = scale * 1.5
		else:
			fig_size_x = scale * 1.5
			fig_size_y = scale

		gdf = gdf.to_crs(epsg=3857)

		fig, ax = plt.subplots(figsize=(fig_size_x, fig_size_y))
		if parameter is not None:
			param_values = self.df_norm[parameter]
			norm = mcolors.Normalize(vmin=min(param_values), vmax=max(param_values))
			scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
			track_color = [mcolors.to_hex(scalar_map.to_rgba(param_values[i]))
							for i in range(len(self.df))]
			ax.scatter(gdf.geometry.x, gdf.geometry.y, color=track_color, s=3.0)
		else:
			ax.plot(gdf.geometry.x, gdf.geometry.y, color='r', markersize=5, alpha=0.7)

		ax.set_yticks([])
		ax.set_xticks([])
		ax.set_xlim(gdf.geometry.x.min() - 1e3, gdf.geometry.x.max() + 1e3)
		ax.set_ylim(gdf.geometry.y.min() - 1e3, gdf.geometry.y.max() + 1e3)

		ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)

		if filename is not None:
			fig.savefig(filename)
		plt.show()

	def plot_parameter(self, parameter, x="dist", filename=None, color='b'):
		columns = list(self.df.columns)

		if parameter not in columns:
			raise Exception("Enter a valid parameter.")
		if x not in columns:
			raise Exception("Enter a valid value for x.")
		if x == 'time':
			raise Exception("Enter a valid value for x. Maybe try with 'time_acc'.")
		
		fig, ax = plt.subplots(1, 1, figsize=(7, 5))
		ax.plot(self.df[x], self.df[parameter], color=color)
		if parameter == 'pace':
			ax.set_ylim(self.df[parameter].max(), self.df[parameter].min() - 1)
		else:
			ax.set_ylim(self.df[parameter].min() - 1, self.df[parameter].max() + 1)
		ax.set_title(f"{self.names[parameter]} vs {self.names[x]}")
		if self.units[x] is not None:
			ax.set_xlabel(f"{self.names[x]} [{self.units[x]}]")
		else:
			ax.set_xlabel(f"{self.names[x]}")
		if self.units[parameter] is not None:
			ax.set_ylabel(f"{self.names[parameter]} [{self.units[parameter]}]")
		else:
			ax.set_ylabel(f"{self.names[parameter]}")

		if filename is not None:
			fig.savefig(filename)
		plt.show()

	def __plot_extra_params(self, ax, x, param, focused_param, colors, handles, i):
		ax2 = ax.twinx()
		handles.append(
			ax2.plot(
				self.df[x],
				self.df_norm[param],
				color=colors[(i) % len(colors)],
				label=self.names[param]
			)[0]
			)
		ax2.set_yticks([])

		if param == focused_param:
			y_max = int(self.df[param].max())
			y_min = int(self.df[param].min())
			if param == 'pace':
				yticks_labels = [str(int(i)) for i in np.linspace(y_max, y_min, 6)]
			else:
				yticks_labels = [str(int(i)) for i in np.linspace(y_min, y_max, 6)]
			y_max = self.df_norm[param].max()
			y_min = self.df_norm[param].min()
			yticks = [i for i in np.linspace(y_min, y_max, 6)]
			ax2.set_yticks(ticks=yticks, labels=yticks_labels)
			ax2.set_ylabel(f"{self.names[focused_param]} [{self.units[focused_param]}]")

		return handles
		



	def plot_parameters(self, *argv, x='dist', fst_param='ele', focused_param=None, filename=None, cmap='gist_rainbow'):

		if len(argv) == 0:
			raise Exception("Insert at least a parameter to plot.")

		if focused_param is None:
			focused_param = argv[-1]

		columns = self.df.columns
		if fst_param not in columns:
			raise Exception(f"{fst_param} is not a valid parameter. Try with 'ele', 'hr', 'cad' or 'pace'.")
		elif fst_param in argv:
			raise Exception(f"'{fst_param}' as fst_param is not a valid because it's already included within the first arguments.")
		elif focused_param not in columns:
			raise Exception(f"{focused_param} is not a valid parameter. Try with 'ele', 'hr', 'cad' or 'pace'.")
		for param in argv:
			if param not in columns:
				raise Exception(f"{param} is not a valid parameter. Try with 'ele', 'hr', 'cad' or 'pace'.")
		if x not in columns:
			raise Exception("Enter a valid value for x.")
		if x == 'time':
			raise Exception("Enter a valid value for x. Maybe try with 'time_acc'.")

		norm = mcolors.Normalize(vmin=0, vmax=1)
		scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
		color_idxs = np.linspace(0, 1, len(argv) + 1)
		colors = [scalar_map.to_rgba(i) for i in color_idxs]

		fig, ax = plt.subplots(1, 1, figsize=(10, 5))

		fst_param_norm = self.df_norm[fst_param]
		ax.set_xlabel(f"{self.names[x]} [{self.units[x]}]")
		ax.set_ylabel(f"{self.names[fst_param]} [{self.units[fst_param]}]")
		y_min = int(self.df[fst_param].min())
		y_max = int(self.df[fst_param].max())
		yticks = [fst_param_norm.min(), fst_param_norm.max()]
		yticks_labels = [str(y_min), str(y_max)]
		ax.fill_between(self.df[x], fst_param_norm.min() * 10, fst_param_norm, color='k', alpha=.5)
		handles = [ax.plot(self.df[x], fst_param_norm, color='k', alpha=.5, linewidth=1, label=self.names[fst_param])[0]]
		ax.axhline(y=fst_param_norm.min(), color='k', linestyle=":", alpha=0.5)
		ax.axhline(y=fst_param_norm.max(), color='k', linestyle=":", alpha=0.5)
		ax.set_yticks(ticks=yticks, labels=yticks_labels)
		ax.set_ylim(fst_param_norm.min() * 10, fst_param_norm.max() * 2)

		for i in range(0, len(argv)):
			handles = self.__plot_extra_params(ax, x, argv[i], focused_param, colors, handles, i)
		

		fig.legend(handles=handles, bbox_to_anchor=(0.85, len(handles) * 0.09), ncol=1, frameon=False)

		if filename is not None:
			fig.savefig(filename)
		plt.show()

	def open_google_maps(self, name=None, parameter=None, cmap="viridis", zoom_start=15, map_type="s", color="blue"):

		track_points = [(self.df['lat'][i], self.df['lon'][i]) for i in range(len(self.df))]
		x_diff = (self.df['lat'].max() - self.df['lat'].min()) / 2
		y_diff = (self.df['lon'].max() - self.df['lon'].min()) / 2
		center = [self.df['lat'].min() + x_diff, self.df['lon'].min() + y_diff]
		m = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
		folium.TileLayer(
        	tiles="http://{s}.google.com/vt/lyrs=" + f"{map_type}" + "&x={x}&y={y}&z={z}",
        	attr="Google Maps",
        	name="Google Maps",
        	subdomains=["mt0", "mt1", "mt2", "mt3"],
    	).add_to(m)

		if parameter is not None:
			param_values = self.df[parameter]
			norm = mcolors.Normalize(vmin=min(param_values), vmax=max(param_values))
			scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
			track_color = [mcolors.to_hex(scalar_map.to_rgba(param_values[i]))
				  			for i in range(len(self.df))]
			for i in range(len(self.df) - 1):
				segment = [track_points[i], track_points[i + 1]]
				folium.PolyLine(segment, color=track_color[i], weight=3, opacity=0.8).add_to(m)
		else:
			folium.PolyLine(track_points, color=color, weight=3, opacity=0.8).add_to(m)
		if name is not None:
			m.save(name + ".html")
			webbrowser.open(name + ".html")
		else:
			m.save(self.gpx_filename + ".html")
			webbrowser.open(self.gpx_filename + ".html")


class LinearPredictor(gpxAnalyseClass):

	def __init__(self, gpx_filename, parameter='hr'):
		gpxAnalyseClass.__init__(self, gpx_filename)
		self.parameter = parameter

	def __from_df_to_ds(self, data_df, label_df, shuffle=True, batch_size=32):
		ds = tf.data.Dataset.from_tensor_slices((data_df.values, label_df.values))
		if shuffle:
			ds = ds.shuffle(buffer_size=len(data_df))
		ds = ds.batch(batch_size)
		return ds

	def __define_train_and_val(self, train_size=0.8, df_norm=None):
		if df_norm is None:
			df = self.df_norm.copy()
		else:
			df = df_norm.copy()
		df = df.drop(['lat', 'lon'], axis=1)
		df_train = df.iloc[:int(len(df) * train_size)]
		df_val = df.iloc[int(len(df) * train_size):]
		y_train = df_train.pop(self.parameter)
		y_val = df_val.pop(self.parameter)
		return df_train, y_train, df_val, y_val

	def __create_model(self, initial_lr=0.01, df_norm=None):
		df_train = self.__define_train_and_val(df_norm=df_norm)[0]
		num_features = len(df_train.columns)
		inputs = tf.keras.layers.Input(shape=(num_features,))
		output = tf.keras.layers.Dense(1)(inputs)
		model = tf.keras.Model(inputs=inputs, outputs=output)
		optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
		model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
		
		return model
	
	def train_model(self, verbose=0, df_norm=None, initial_lr=0.01, batch_size=32, patience_lr_change=50, patience_early_stop=100, epochs=200):
		model = self.__create_model(initial_lr=initial_lr, df_norm=df_norm)
		df_train, y_train, df_val, y_val = self.__define_train_and_val(df_norm=df_norm)
		ds_train = self.__from_df_to_ds(df_train, y_train, batch_size=batch_size)
		ds_val = self.__from_df_to_ds(df_val, y_val, shuffle=False, batch_size=batch_size)
		callbacks = [
			tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience_lr_change, min_lr=1e-6),
			tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_early_stop, restore_best_weights=True)
		]
		model_history = model.fit(ds_train, validation_data=ds_val, verbose=verbose, epochs=epochs, callbacks=callbacks)
		return model, model_history
	
	def plot_performance(self, model_history, filename=None):
		fig, ax = plt.subplots(1, 2, figsize=(15, 5))
		ax[0].plot(model_history.history['loss'], label='loss', color='b')
		ax[0].plot(model_history.history['val_loss'], label='val_loss', color='r')
		ax[0].set_title('Loss and Validation Loss curves')
		ax[0].set_xlabel('Epochs')
		ax[0].set_ylabel('Loss')
		ax[0].legend()

		ax[1].plot(model_history.history['mae'], label='mae', color='b')
		ax[1].plot(model_history.history['val_mae'], label='val_mae', color='r')
		ax[1].set_title('Mean Absolute Error and Validation MAE curves')
		ax[1].set_xlabel('Epochs')
		ax[1].set_ylabel('MAE')
		ax[1].legend()

		if filename is not None:
			fig.savefig(filename)
		plt.show()
	
	def back_to_original(self, df, mean, std):
		return df * std + mean

	def predictions(self, model, df_test_norm):
		df_test = df_test_norm.drop(['lat', 'lon'], axis=1)
		y_test = df_test.pop(self.parameter)
		ds_test = self.__from_df_to_ds(df_test, y_test, shuffle=False)
		predictions = model.predict(ds_test)
		return self.back_to_original(predictions, self.df[self.parameter].mean(), self.df[self.parameter].std())
	
	def plot_predictions(self, model, df_test_norm, df_test, x='dist', filename=None):
		predictions = self.predictions(model, df_test_norm)
		y_test = df_test[self.parameter]
		x_values = df_test[x]

		fig, ax = plt.subplots(1, 1, figsize=(10, 5))
		ax.scatter(x_values, y_test, s=5, color='b', label='Real values')
		ax.plot(x_values, predictions, color='r', label='Predictions')
		ax.set_title(f'Real values vs Predictions for {self.names[self.parameter]}')
		ax.set_xlabel(f'{self.names[x]} [{self.units[x]}]')
		ax.set_ylabel(f'{self.names[self.parameter]} [{self.units[self.parameter]}]')
		ax.legend()

		if filename is not None:
			fig.savefig(filename)
		plt.show()
