
import gpxpy as gpx
import geopandas as gpd
import contextily as ctx
import folium
import webbrowser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class gpxAnalyserClass:

	units: dict = {
		'lat': None,
		'lon': None,
		'ele': "m",
		'time': None,
		'hr': "beats/min",
		'cad': "steps/min",
		'dist': "m",
		'time_acc': "min",
		'speed': "min/Km"
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
		'speed': "Velocity"
	}

	def __init__(self, gpx_filename):
		with open(gpx_filename, "r") as gpx_file:
			gpx_object = gpx.parse(gpx_file)
		self.gpx_filename = gpx_filename.split('.')[0]
		self.df = self.gpx_to_df(gpx_object)
		self.df['cad'] = self.df['cad'] + 100
		self.df['dist'] = self.get_distances()
		self.df['time_acc'] = self.get_times()
		self.df['speed'] = self.get_speeds()
		for i in range(len(self.df)):
			if self.df['speed'][i] > 20:
				self.df.loc[i, 'speed'] = 20

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

	def plot_track_satelite(self, filename=None):
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
		ax.plot(gdf.geometry.x, gdf.geometry.y, color='red', markersize=5, alpha=0.7)

		#ax.set_xlabel(f"{self.names['lon']}")
		#ax.set_ylabel(f"{self.names['lat']}")
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
		if parameter == "speed":
			ax.set_ylim(self.df[parameter].max(), self.df[parameter].min() - 1)
		else:
			ax.set_ylim(self.df[parameter].min() - 1, self.df[parameter].max() + 1)
		ax.set_title(f"{parameter} vs {x}")
		if self.units[x] is not None:
			ax.set_xlabel(f"{x} [{self.units[x]}]")
		else:
			ax.set_xlabel(f"{x}")
		if self.units[parameter] is not None:
			ax.set_ylabel(f"{parameter} [{self.units[parameter]}]")
		else:
			ax.set_ylabel(f"{parameter}")

		if filename is not None:
			fig.savefig(filename)
		plt.show()

	def plot_different_param(self, *argv, x='dist', fst_param='ele', filename=None, colors=['b', 'r', 'orange', 'cyan', 'magenta']):

		if len(argv) == 0:
			raise Exception("Insert at least two parameters to plot.")

		fig, ax = plt.subplots(1, 1, figsize=(7, 4))

		fst_param_data = self.df[fst_param]
		ax.set_xlabel(f"{x} [{self.units[x]}]")
		y_min = int(fst_param_data.min())
		y_max = int(fst_param_data.max())
		yticks = [y_min / y_max, 1]
		yticks_labels = [str(y_min), str(y_max)]
		handles = [ax.plot(self.df[x], fst_param_data / y_max, color=colors[0], label=fst_param)[0]]
		ax.axhline(y=y_min / y_max, color='k', linestyle=":", alpha=0.5)
		ax.axhline(y=1, color='k', linestyle=":", alpha=0.5)
		ax.set_yticks(ticks=yticks, labels=yticks_labels)

		for i in range(0, len(argv)):

			y_max = int(self.df[argv[i]].max())
			if i != (len(argv) - 1):
				handles.append(ax.plot(self.df[x], self.df[argv[i]] / y_max,
						   color=colors[(i + 1) % len(colors)], label=argv[i])[0])
			else:
				ax2 = ax.twinx()
				y_min = int(self.df[argv[i]].min())
				y_steps = int((y_max - y_min) / 5)
				yticks = [i / y_max for i in range(y_min, y_max, y_steps)]
				yticks_labels = [str(i) for i in range(y_min, y_max, y_steps)]
				handles.append(ax2.plot(self.df[x], self.df[argv[i]] / y_max,
							color=colors[(i + 1) % len(colors)], label=argv[i])[0])
				ax2.set_yticks(ticks=yticks, labels=yticks_labels)
		

		fig.legend(handles=handles, bbox_to_anchor=(0.90, len(handles) * 0.1), ncol=1, frameon=False)

		if filename is not None:
			fig.savefig(filename)
		plt.show()

	def open_google_maps(self, name=None, parameter=None, cmap="magma", zoom_start=15, map_type="s", color="blue"):

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