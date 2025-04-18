#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Frank Asto
## DS 260
## Project
## Water use in Groundwater Management Districts (1, 3, and 4)
import os, warnings, zipfile
import gdown
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from shapely.geometry import Point
from IPython.display import display
warnings.filterwarnings("ignore")

# In[577]:

# ---------------------------------------------------------------------------
# Google‑Drive helper – download only once per session
# ---------------------------------------------------------------------------
DRIVE_IDS = {
    "main"  : "1wqUxTFOtv2C2LlxiqJN_49kGDdKwXUNI",
    "wizard": "1oS3c3qzMRgpYA05NkB7LUULLLrsZsVxM",
    "wimas" : "1i9ODCyZ32tLprxeSSbSjcwjVXSjyVYGH",
    "huc10" : "1Y2rx3Cq-pidPfaDZ3hAvC2qWJCimqkGk",
    "gmd"   : "1EqOiQ-7fjVQiZci3XZ5ej1S5-0dpOTc9",
}

def fetch(file_key: str, dest: str) -> str:
    """Download Google‑Drive file if not present and return local path."""
    if not os.path.exists(dest):
        url = f"https://drive.google.com/uc?id={DRIVE_IDS[file_key]}"
        gdown.download(url, dest, quiet=False)
    return dest

# ---------------------------------------------------------------------------
# 1) Load the three water‑use CSVs
# ---------------------------------------------------------------------------
main_csv   = fetch("main",   "GMD_water_use.csv")
wizard_csv = fetch("wizard", "wizard.csv")
wimas_csv  = fetch("wimas",  "wimas.csv")

print("Loading large CSVs …")
df_main   = pd.read_csv(main_csv)
df_wizard = pd.read_csv(wizard_csv)
df_wimas  = pd.read_csv(wimas_csv)

print("Loaded shapes:")
print("• df_main  :",   df_main.shape)
print("• df_wizard:", df_wizard.shape)
print("• df_wimas :", df_wimas.shape)

#size of rows and columns
print("Total Rows and Columns in GMD Water file:")
print(df_main.shape)

print("\nColumn Names: \n")
# Print all column names
for col in df_main.columns:
    print(col)

#size of rows and columns
print("\nTotal Rows and Columns in Wizard file:")
print(df_wizard.shape)

print("\nColumn Names: \n")
# Print all column names
for col in df_wizard.columns:
    print(col)

#size of rows and columns
print("\nTotal Rows and Columns in Wimas file:")
print(df_wimas.shape)

print("\nColumn Names: \n")
# Print all column names
for col in df_wimas.columns:
    print(col)


# In[579]:


# Make a copy of the full dataset
df_main_cols = df_main.copy()

# Drop columns I don't need
static_columns_to_drop = [
    'county_abrev', 'aquier_codes', 'COUSUB_NAME', 'COUSUB_GEOID',
    'PLSS_ID', 'PDIV_ID', 'OID_', 'source', 'active_20230507', 
]

columns_to_drop = []
for col in static_columns_to_drop:
    if col in df_main_cols.columns:
        columns_to_drop.append(col)

df_main_cols.drop(columns=columns_to_drop, inplace=True)

# Drop ACRES columns (1990–2022)
acres_columns_to_drop = []
for year in range(1990, 2023):
    col_name = f'ACRES_{year}'
    if col_name in df_main_cols.columns:
        acres_columns_to_drop.append(col_name)

df_main_cols.drop(columns=acres_columns_to_drop, inplace=True)

# Preview cleaned data
df_main_cols.head()


# In[581]:


# Print data type of df_main_cols
pd.set_option('display.max_rows', None)
print(df_main_cols.dtypes)


# In[583]:


#checking for null values in df_main_cols
print(df_main_cols.isnull().sum())


# In[585]:


# I have 17256 missing values for GMD, but I have the County Names, I'll make a dictionary to map it with the missing district

county_to_gmd = {
    'Allen': 3, 'Anderson': 3, 'Atchison': 3, 'Barber': 1, 'Barton': 4,
    'Bourbon': 3, 'Brown': 3, 'Butler': 2, 'Chase': 2, 'Chautauqua': 3,
    'Cherokee': 3, 'Cheyenne': 4, 'Clark': 1, 'Clay': 2, 'Cloud': 2,
    'Coffey': 3, 'Comanche': 1, 'Cowley': 2, 'Crawford': 3, 'Decatur': 4,
    'Dickinson': 2, 'Doniphan': 3, 'Douglas': 3, 'Edwards': 1, 'Elk': 3,
    'Ellis': 4, 'Ellsworth': 2, 'Finney': 1, 'Ford': 1, 'Franklin': 3,
    'Geary': 2, 'Gove': 4, 'Graham': 4, 'Grant': 1, 'Gray': 1,
    'Greeley': 1, 'Greenwood': 3, 'Hamilton': 1, 'Harper': 2, 'Harvey': 2,
    'Haskell': 1, 'Hodgeman': 1, 'Jackson': 3, 'Jefferson': 3, 'Jewell': 2,
    'Johnson': 3, 'Kearny': 1, 'Kingman': 2, 'Kiowa': 1, 'Labette': 3,
    'Lane': 1, 'Leavenworth': 3, 'Lincoln': 2, 'Linn': 3, 'Logan': 4,
    'Lyon': 3, 'Mcpherson': 2, 'McPherson': 2, 'Marion': 2, 'Marshall': 2, 'Meade': 1,
    'Miami': 3, 'Mitchell': 2, 'Montgomery': 3, 'Morris': 2, 'Morton': 1,
    'Nemaha': 3, 'Neosho': 3, 'Ness': 1, 'Norton': 4, 'Osage': 3,
    'Osborne': 4, 'Ottawa': 2, 'Pawnee': 1, 'Phillips': 4, 'Pottawatomie': 3,
    'Pratt': 1, 'Rawlins': 4, 'Reno': 2, 'Republic': 2, 'Rice': 2,
    'Riley': 2, 'Rooks': 4, 'Rush': 1, 'Russell': 4, 'Saline': 2,
    'Scott': 1, 'Sedgwick': 2, 'Seward': 1, 'Shawnee': 3, 'Sheridan': 4,
    'Sherman': 4, 'Smith': 4, 'Stafford': 1, 'Stanton': 1, 'Stevens': 1,
    'Sumner': 2, 'Thomas': 4, 'Trego': 4, 'Wabaunsee': 3, 'Wallace': 4,
    'Washington': 2, 'Wichita': 1, 'Wilson': 3, 'Woodson': 3, 'Wyandotte': 3
}


# In[587]:


# Split out the two subsets
has_gmd    = df_main_cols[df_main_cols['gmd'].notna()].copy()
missing_gmd = df_main_cols[df_main_cols['gmd'].isna()].copy()

# Fill the missing ones
missing_gmd['gmd'] = missing_gmd.apply(
    lambda row: county_to_gmd.get(row['COUNTY_NAME'], row['gmd']),
    axis=1)

# Re‑combine
df_main_cols = pd.concat([has_gmd, missing_gmd], ignore_index=True)


# In[589]:


# this is again to check for missing values, 3 for gmd and 7 for county names, so I'll drop them
pd.set_option('display.max_rows', None)
print(df_main_cols.isnull().sum())


# In[591]:


# removing the missing data from the columns
df_main_cols = df_main_cols.dropna(subset=['gmd', 'COUNTY_NAME', 'COUNTY_GEOID'])


# In[593]:


# check again
pd.set_option('display.max_rows', None)
print(df_main_cols.isnull().sum())


# In[595]:


#check unique GMD - districts
df_main_cols["gmd"].unique()


# In[597]:


#keeping only 1, 3, 4 on GMD
df_main_cols = df_main_cols[df_main_cols['gmd'].isin([1, 3, 4])]


# In[599]:


#check again unique GMD - districts
df_main_cols["gmd"].unique()


# In[601]:


#checking count of rows in dataframe
print("Total Rows and Columns in GMD Water file:")
print(df_main_cols.shape)


# In[603]:


#remove duplicates on df_main_cols
df_main_cols = df_main_cols.drop_duplicates()


# In[605]:


#checking count of rows in dataframe
print("Total Rows and Columns in GMD Water file:")
print(df_main_cols.shape)


# In[607]:


# Creating a simple line chart with average water use over time in all districts (1, 3, and 4)
import matplotlib.pyplot as plt

# Step 1: Get all AF_USED columns
af_columns = [col for col in df_main_cols.columns if col.startswith("AF_USED_") and col[8:].isdigit()]

# Step 2: Extract years from the column names for x-axis
years = [int(col.split("_")[-1]) for col in af_columns]

# Step 3: Calculate the mean per year
avg_af_per_year = df_main_cols[af_columns].mean()

# Step 4: Plot with years on x-axis
plt.figure(figsize=(10, 5))
plt.plot(years, avg_af_per_year, marker="o")
plt.title("Average Water Use Over Time In All The Districts (1, 3, and 4)")
plt.xlabel("Year")
plt.ylabel("Acre-Feet (AF)")
plt.grid(True)
plt.tight_layout()
plt.show()



# In[611]:


import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

# Define a dictionary for use types and their column prefixes
use_types = {
    "Total": "AF_USED_",
    "Irrigation": "AF_USED_IRR_",
    "Municipal": "AF_USED_MUN_",
    "Stock": "AF_USED_STK_",
    "Industrial": "AF_USED_IND_",
    "Recreational": "AF_USED_REC_"
}

# Interactive function
def plot_gmd_use(gmd_number, use_type):
    prefix = use_types[use_type]
    af_columns = [col for col in df_main_cols.columns if col.startswith(prefix) and col[len(prefix):].isdigit()]
    years = [int(col.split('_')[-1]) for col in af_columns]

    if gmd_number == "All":
        # Plot all GMDs together
        df_list = []
        for gmd in [1, 3, 4]:
            gmd_df = df_main_cols[df_main_cols['gmd'] == gmd]
            avg_af = gmd_df[af_columns].mean()
            df_temp = pd.DataFrame({
                'Year': years,
                'Avg_AF_Used': avg_af.values,
                'GMD': f'GMD {gmd}'
            })
            df_list.append(df_temp)

        df_plot = pd.concat(df_list, ignore_index=True)
        fig = px.line(df_plot, x='Year', y='Avg_AF_Used', color='GMD',
                      title=f'{use_type} Water Use Over Time (GMDs 1, 3, 4)', markers=True)
        max_y = df_plot['Avg_AF_Used'].max()

    else:
        # Single GMD
        gmd_df = df_main_cols[df_main_cols['gmd'] == gmd_number]
        avg_af = gmd_df[af_columns].mean()
        df_plot = pd.DataFrame({
            'Year': years,
            'Avg_AF_Used': avg_af.values
        })
        fig = px.line(df_plot, x='Year', y='Avg_AF_Used',
                      title=f'{use_type} Water Use Over Time (GMD {gmd_number})', markers=True)
        max_y = df_plot['Avg_AF_Used'].max()

    fig.update_layout(
        height=600,
        xaxis_title='Year',
        yaxis_title='Acre-Feet Used',
        yaxis=dict(range=[0, max_y * 1.2])
    )
    fig.show()

# Create widgets
gmd_dropdown = widgets.Dropdown(
    options=[("GMD 1", 1), ("GMD 3", 3), ("GMD 4", 4), ("All GMDs", "All")],
    value=1,
    description='Select GMD:'
)

use_radio = widgets.RadioButtons(
    options=list(use_types.keys()),
    value="Total",
    description='Use Type:',
    layout={'width': 'max-content'}
)

# Display both widgets
ui = widgets.HBox([gmd_dropdown, use_radio])
out = widgets.interactive_output(plot_gmd_use, {'gmd_number': gmd_dropdown, 'use_type': use_radio})

display(ui, out)


# In[615]:


# Creating a bar chart for the GMDs with categories and a Play animation for the years

# Imports
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

# Define use types and their column prefixes
use_types = {
    "Total":       "AF_USED_",
    "Irrigation":  "AF_USED_IRR_",
    "Municipal":   "AF_USED_MUN_",
    "Stock":       "AF_USED_STK_",
    "Industrial":  "AF_USED_IND_",
    "Recreational":"AF_USED_REC_"
}

# Derive years from the Total prefix
total_prefix = use_types['Total']
years = sorted(
    int(c.split('_')[-1])
    for c in df_main_cols.columns
    if c.startswith(total_prefix) and c[len(total_prefix):].isdigit()
)

# GMD selection dropdown
gmd_dropdown = widgets.Dropdown(
    options=[('GMD 1',1), ('GMD 3',3), ('GMD 4',4), ('All GMDs','All')],
    value=1,
    description='GMD:'
)

# Year slider and play button for animation
year_slider = widgets.IntSlider(
    value=years[0],
    min=years[0],
    max=years[-1],
    step=1,
    description='Year:'
)
play_button = widgets.Play(
    value=years[0],
    min=years[0],
    max=years[-1],
    step=1,
    interval=400,  # speed of animation
    description='Play'
)
# Link play button and slider
widgets.jslink((play_button, 'value'), (year_slider, 'value'))

# Update function for animated bar chart
def update_bar_chart(gmd_selected, year_selected):
    data = []
    gmd_list = [1, 3, 4] if gmd_selected == 'All' else [gmd_selected]

    for gmd in gmd_list:
        df_g = df_main_cols[df_main_cols['gmd'] == gmd]
        for category, prefix in use_types.items():
            # find relevant columns for this category-year
            col_year = f"{prefix}{year_selected}"
            val = df_g[col_year].mean() if col_year in df_g.columns else 0
            data.append({
                'GMD': f'GMD {gmd}',
                'Category': category,
                'Avg_AF_Used': val
            })

    # Build DataFrame and sort descending by value
    df_bar = pd.DataFrame(data).sort_values('Avg_AF_Used', ascending=False)

    # Create horizontal bar chart
    fig = px.bar(
        df_bar,
        x='Avg_AF_Used',
        y='Category',
        orientation='h',
        color='Avg_AF_Used',
        color_continuous_scale=['lightblue', 'darkblue'],
        facet_col='GMD',
        category_orders={'Category': df_bar['Category'].tolist()},
        title=f"Category Water Use in {year_selected} by GMD"
    )
    fig.update_layout(
        xaxis_title='Average Acre‑Feet Used',
        yaxis_title='',
        height=500,
        margin={'t': 50, 'b': 40}
    )
    fig.show()

# Display interactive UI with animation controls
ui = widgets.HBox([gmd_dropdown, play_button, year_slider])
out = widgets.interactive_output(
    update_bar_chart,
    {'gmd_selected': gmd_dropdown, 'year_selected': year_slider}
)

display(ui, out)


# In[617]:


df_wizard.head()


# In[619]:


# Select only the needed columns from the original df_wizard
df_wizard_cols = df_wizard[['wellid', 'date', 'depth', 'lat', 'long', 'countcode']].copy()
df_wizard_cols['gmd'] = None
df_wizard_cols['COUNTY_NAME'] = None


# In[621]:


# Print data type of df_wizard_cols_cols
pd.set_option('display.max_rows', None)
print(df_wizard_cols.dtypes)


# In[623]:


print("Total Rows and Columns in Wizard file:")
print(df_wizard_cols.shape)


# In[625]:


#remove duplicates on df_wizard_cols
df_wizard_cols = df_wizard_cols.drop_duplicates()


# In[627]:


print("Total Rows and Columns in Wizard file:")
print(df_wizard_cols.shape)


# In[629]:


# Convert date column to datetime
df_wizard_cols['date'] = pd.to_datetime(df_wizard_cols['date'], errors='coerce')

# Extract year from date
df_wizard_cols['year'] = df_wizard_cols['date'].dt.year

# Step 4: Map countcode to COUNTY_NAME
countcode_to_county = {
    1: 'Allen', 3: 'Anderson', 5: 'Atchison', 7: 'Barber', 9: 'Barton', 11: 'Bourbon', 13: 'Brown',
    15: 'Butler', 17: 'Chase', 19: 'Chautauqua', 21: 'Cherokee', 23: 'Cheyenne', 25: 'Clark', 27: 'Clay',
    29: 'Cloud', 31: 'Coffey', 33: 'Comanche', 35: 'Cowley', 37: 'Crawford', 39: 'Decatur', 41: 'Dickinson',
    43: 'Doniphan', 45: 'Douglas', 47: 'Edwards', 49: 'Elk', 51: 'Ellis', 53: 'Ellsworth', 55: 'Finney',
    57: 'Ford', 59: 'Franklin', 61: 'Geary', 63: 'Gove', 65: 'Graham', 67: 'Grant', 69: 'Gray', 71: 'Greeley',
    73: 'Greenwood', 75: 'Hamilton', 77: 'Harper', 79: 'Harvey', 81: 'Haskell', 83: 'Hodgeman', 85: 'Jackson',
    87: 'Jefferson', 89: 'Jewell', 91: 'Johnson', 93: 'Kearny', 95: 'Kingman', 97: 'Kiowa', 99: 'Labette',
    101: 'Lane', 103: 'Leavenworth', 105: 'Lincoln', 107: 'Linn', 109: 'Logan', 111: 'Lyon', 113: 'McPherson',
    115: 'Marion', 117: 'Marshall', 119: 'Meade', 121: 'Miami', 123: 'Mitchell', 125: 'Montgomery',
    127: 'Morris', 129: 'Morton', 131: 'Nemaha', 133: 'Neosho', 135: 'Ness', 137: 'Norton', 139: 'Osage',
    141: 'Osborne', 143: 'Ottawa', 145: 'Pawnee', 147: 'Phillips', 149: 'Pottawatomie', 151: 'Pratt',
    153: 'Rawlins', 155: 'Reno', 157: 'Republic', 159: 'Rice', 161: 'Riley', 163: 'Rooks', 165: 'Rush',
    167: 'Russell', 169: 'Saline', 171: 'Scott', 173: 'Sedgwick', 175: 'Seward', 177: 'Shawnee', 179: 'Sheridan',
    181: 'Sherman', 183: 'Smith', 185: 'Stafford', 187: 'Stanton', 189: 'Stevens', 191: 'Sumner', 193: 'Thomas',
    195: 'Trego', 197: 'Wabaunsee', 199: 'Wallace', 201: 'Washington', 203: 'Wichita', 205: 'Wilson',
    207: 'Woodson', 209: 'Wyandotte'
}

df_wizard_cols['COUNTY_NAME'] = df_wizard_cols['countcode'].map(countcode_to_county)

# View the cleaned result
df_wizard_cols.head()


# In[631]:


# Adding the values of gmd by mapping the county with the district code
df_wizard_cols['gmd'] = df_wizard_cols['COUNTY_NAME'].map(county_to_gmd)

# View the cleaned result
df_wizard_cols.head()


# In[633]:


df_wizard_cols["gmd"].unique()


# In[635]:


#keeping only 1, 3, 4 on GMD
df_wizard_cols = df_wizard_cols[df_wizard_cols['gmd'].isin([1, 3, 4])]


# In[637]:


df_wizard_cols["gmd"].unique()


# In[639]:


pd.set_option('display.max_rows', None)
print(df_wizard_cols.isnull().sum())


# In[641]:


import matplotlib.pyplot as plt

# Step 1: Filter data to include only years from 1940 and onward
depth_by_year_filtered = df_wizard_cols[df_wizard_cols['year'] >= 1940]

# Step 2: Group by year and calculate the average depth
depth_by_year_avg = depth_by_year_filtered.groupby('year')['depth'].mean()

# Step 3: Plot the results with reversed y-axis
plt.figure(figsize=(10, 5))
plt.plot(depth_by_year_avg.index, depth_by_year_avg.values, marker='o', linestyle='-')
plt.title("Average Water Depth Over Time (Starting 1935)")
plt.xlabel("Year")
plt.ylabel("Average Depth (feet)")
plt.gca().invert_yaxis()  # Reverses the y-axis
plt.grid(True)
plt.tight_layout()
plt.show()



# In[645]:


import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

# Step 1: Map COUNTY_NAME to GMD using df_main_cols
county_to_gmd = df_main_cols[['COUNTY_NAME', 'gmd']].drop_duplicates().set_index('COUNTY_NAME')['gmd'].to_dict()
df_wizard_cols['gmd'] = df_wizard_cols['COUNTY_NAME'].map(county_to_gmd)

# Step 2: Filter WIZARD data: years >= 1940 and GMDs 1, 3, 4
df_wizard_filtered = df_wizard_cols[
    (df_wizard_cols['year'] >= 1940) &
    (df_wizard_cols['gmd'].isin([1, 3, 4]))
]

# Step 3: Define plot function
def plot_depth_by_gmd(gmd_selection):
    if gmd_selection == "All":
        fig_df = df_wizard_filtered.groupby(['year', 'gmd'])['depth'].mean().reset_index()
        fig_df['GMD'] = fig_df['gmd'].apply(lambda x: f'GMD {int(x)}')
        fig = px.line(
            fig_df, x='year', y='depth', color='GMD',
            title='Average Water Depth Over Time by GMD (Starting 1940)',
            markers=True
        )
    else:
        gmd_num = int(gmd_selection)
        gmd_df = df_wizard_filtered[df_wizard_filtered['gmd'] == gmd_num]
        avg_depth = gmd_df.groupby('year')['depth'].mean().reset_index()
        fig = px.line(
            avg_depth, x='year', y='depth',
            title=f'Average Water Depth Over Time - GMD {gmd_num} (Starting 1935)',
            markers=True
        )

    fig.update_layout(
        height=600,
        xaxis_title='Year',
        yaxis_title='Average Depth (feet)',
        yaxis=dict(autorange='reversed'),  # Shallower appears higher
        xaxis=dict(dtick=5)
    )
    fig.show()

# Step 4: Dropdown for GMD selection
gmd_dropdown = widgets.Dropdown(
    options=[("GMD 1", "1"), ("GMD 3", "3"), ("GMD 4", "4"), ("All GMDs", "All")],
    value="1",
    description="Select GMD:"
)

# Step 5: Show dropdown + plot
widgets.interact(plot_depth_by_gmd, gmd_selection=gmd_dropdown)


# In[647]:


# Print wimas columns, for this file I am going to use all the columns
df_wimas_cols = df_wimas.copy()
df_wimas_cols.head()


# In[649]:


# Print data type of df_wimas_cols
pd.set_option('display.max_rows', None)
print(df_wimas_cols.dtypes)


# In[651]:


#checking for null values in df_wimas_cols
print(df_wimas_cols.isnull().sum())


# In[653]:


#remove duplicates on df_wimas_cols
df_wimas_cols = df_wimas_cols.drop_duplicates()


# In[655]:


#checking count of rows in dataframe
print("Total Rows and Columns in Wimas file:")
print(df_wimas_cols.shape)


# In[657]:


# Creating a simple line chart with average water use HUC10
import matplotlib.pyplot as plt

# Step 1: Get all AF_USED columns
af_columns = [col for col in df_wimas_cols.columns if col.startswith("AF_")]

# Step 2: Extract years from the column names for x-axis
years = [int(col.split("_")[-1]) for col in af_columns]

# Step 3: Calculate the mean per year
avg_af_per_year = df_wimas_cols[af_columns].mean()

# Step 4: Plot with years on x-axis
plt.figure(figsize=(10, 5))
plt.plot(years, avg_af_per_year, marker="o")
plt.title("Average Water Use HUC10")
plt.xlabel("Year")
plt.ylabel("Acre-Feet (AF)")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[690]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

wimas_df = df_wimas_cols.copy()
wizard_df = df_wizard_cols.copy()

huc10_geo  = fetch("huc10","HUC10.geojson")
gmd_geo    = fetch("gmd",  "GMD.geojson")
huc10_gdf  = gpd.read_file(huc10_geo)
gmd_gdf    = gpd.read_file(gmd_geo)

# Step 2: Ensure CRS matches between HUC10 and GMD
huc10_gdf = huc10_gdf.to_crs(gmd_gdf.crs)

# Step 3: Spatial join - map HUC10s to GMDs
huc10_with_gmd = gpd.sjoin(huc10_gdf, gmd_gdf, how="inner", predicate="intersects")

# Step 4: Clean HUC codes for merging with WIMAS
wimas_df["HUC"] = wimas_df["HUC"].astype(str)
huc10_with_gmd["HUC_10"] = huc10_with_gmd["HUC_10"].astype(str)

# Step 5: Merge WIMAS water use data with HUC10 + GMD regions
merged_huc10 = huc10_with_gmd.merge(wimas_df, left_on="HUC_10", right_on="HUC")

# Step 6: Convert WIZARD wells to GeoDataFrame
wizard_df["geometry"] = wizard_df.apply(lambda row: Point(row["long"], row["lat"]), axis=1)
wizard_gdf = gpd.GeoDataFrame(wizard_df, geometry="geometry", crs="EPSG:4326")
wizard_gdf = wizard_gdf.to_crs(gmd_gdf.crs)

# Step 7: Spatial join - map wells to GMDs using correct GMD_ID
wizard_with_gmd = gpd.sjoin(wizard_gdf, gmd_gdf, how="inner", predicate="within")

# Step 8: Extract year for filtering and analysis
wizard_with_gmd["year"] = pd.to_datetime(wizard_with_gmd["date"]).dt.year


# In[692]:


# Preview merged HUC10 with GMD and water use
merged_huc10.head()


# In[694]:


# Preview wells with GMD and depth
wizard_with_gmd.head()


# In[696]:


# See what GMDs are available
merged_huc10["GMD_ID"].unique(), wizard_with_gmd["GMD_ID"].unique()


# In[698]:


# Imports
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
import json
import warnings
warnings.filterwarnings("ignore")

# Load Groundwater Management District boundaries and prepare GeoJSON
gmd_gdf = gpd.read_file("Groundwater_Management_Districts_(GMD).geojson")
gmd_gdf = gmd_gdf[gmd_gdf["GMD_ID"].isin([1, 3, 4])]
gmd_geojson = json.loads(gmd_gdf.to_json())

# Approximate Kansas boundary via bounding box (lat/lon)
ks_lon_min, ks_lat_min = -102.051, 36.993
ks_lon_max, ks_lat_max = -94.588, 40.003
ks_lons = [ks_lon_min, ks_lon_max, ks_lon_max, ks_lon_min, ks_lon_min]
ks_lats = [ks_lat_min, ks_lat_min, ks_lat_max, ks_lat_max, ks_lat_min]

# === FILTER DATA FOR GMDs 1, 3, 4 ONLY ===
merged_huc10 = merged_huc10[merged_huc10["GMD_ID"].isin([1, 3, 4])]
wizard_with_gmd = wizard_with_gmd[wizard_with_gmd["GMD_ID"].isin([1, 3, 4])]

# Setup year options
years = list(range(1990, 2023))
year_options = ["All Years"] + years

# Widgets
gmd_selector = widgets.Dropdown(options=[1, 3, 4, 'All'], value=1, description="GMD:")
year_selector = widgets.Dropdown(options=year_options, value=2022, description="Year:")
data_selector = widgets.RadioButtons(options=["Water Used", "Water Depth"], value="Water Used", description="Data:")

# Map update function
def update_map(gmd_selected, year_selected, data_type_selected):
    # Select data for HUC10 and wells
    if gmd_selected == 'All':
        df_huc = merged_huc10.copy()
        df_wiz = wizard_with_gmd.copy()
    else:
        df_huc = merged_huc10[merged_huc10["GMD_ID"] == gmd_selected].copy()
        df_wiz = wizard_with_gmd[wizard_with_gmd["GMD_ID"] == gmd_selected].copy()

    # Build figure
    if data_type_selected == "Water Used":
        df = df_huc
        if year_selected != "All Years":
            year_col = f"AF_{year_selected}"
            df = df[df[year_col].notnull()]
            color_col = year_col
        else:
            af_cols = [c for c in df.columns if c.startswith("AF_")]
            df["avg_water_use"] = df[af_cols].mean(axis=1)
            color_col = "avg_water_use"
        fig = px.choropleth_mapbox(
            df,
            geojson=df.geometry.__geo_interface__,
            locations=df.index,
            color=color_col,
            mapbox_style="open-street-map",
            color_continuous_scale=[[0, "yellow"], [1, "darkblue"]],
            center={"lat": 38.5, "lon": -98.5},
            zoom=6,
            opacity=0.6,
            hover_name="HU_10_NAME",
            hover_data={
                "HUC_10": True,
                color_col: True,
                "GMD_ID": True
            },
            title=f"{('All GMDs' if gmd_selected=='All' else f'GMD {gmd_selected}')} - Water Used ({year_selected})"
        )
    else:
        df = df_wiz
        if year_selected != "All Years":
            df = df[df["year"] == int(year_selected)]
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="long",
            color="depth",
            mapbox_style="open-street-map",
            color_continuous_scale="RdBu",
            center={"lat": 38.5, "lon": -98.5},
            zoom=6,
            size_max=12,
            hover_name="wellid",
            hover_data={
                "year": True,
                "depth": True,
                "GMD_ID": True
            },
            title=f"{('All GMDs' if gmd_selected=='All' else f'GMD {gmd_selected}')} - Water Depth ({year_selected})"
        )

    # Highlight GMD boundaries when 'All' is selected
    if gmd_selected == 'All':
        fig.update_layout(
            mapbox_layers=[
                {"source": gmd_geojson, "type": "line", "color": "darkblue", "line": {"width": 3}}
            ]
        )

    # Overlay Kansas state boundary
    fig.add_trace(go.Scattermapbox(
        lon=ks_lons,
        lat=ks_lats,
        mode='lines',
        line=dict(color='black', width=3),
        showlegend=False
    ))

    # Final adjustments
    fig.update_layout(
        height=800,
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    fig.show()

# Launch interactive map
widgets.interact(
    update_map,
    gmd_selected=gmd_selector,
    year_selected=year_selector,
    data_type_selected=data_selector
)


# In[699]:


#comparing Total Average Water use vs Water Depth in Wells
import pandas as pd

# List your years
years = list(range(1990, 2023))

# Compute average water use per year (across all records)
water_use_ave = df_main_cols.copy()
water_use_avg = [
    df_water_use[f"AF_USED_{yr}"].mean()
    for yr in years
]

# Compute average depth per year
wizard1 = df_wizard_cols
wizard1["year"] = pd.to_datetime(
    wizard1["date"], errors="coerce"
).dt.year

depth_ave = df_wizard_cols
depth_avg = [
    wizard1.loc[
        wizard1["year"] == yr, "depth"
    ].mean()
    for yr in years
]

# Build your 3‑column summary
summary_df = pd.DataFrame({
    "year": years,
    "avg_water_use": water_use_avg,
    "avg_depth": depth_avg
})


# In[700]:


import matplotlib.pyplot as plt

# summary_df should already exist with columns 'avg_water_use' and 'avg_depth'
plt.figure(figsize=(8, 6))
plt.scatter(summary_df['avg_water_use'], summary_df['avg_depth'])
plt.xlabel('Average Water Use')
plt.ylabel('Average Water Depth')
plt.title('Scatter Plot: Water Use vs Water Depth')
plt.grid(True)
plt.show()

corr = summary_df["avg_water_use"].corr(summary_df["avg_depth"])
print("Correlation:", corr)


# In[701]:


#comparing Total Average Water use vs Water Depth in Wells
import pandas as pd

# List your years
years = list(range(1990, 2023))

# Compute average water use per year (across all records)
water_use_ave = df_wimas_cols.copy()
water_use_avg = [
    df_wimas_cols[f"AF_{yr}"].mean()
    for yr in years
]

# Compute average depth per year
wizard1 = df_wizard_cols
wizard1["year"] = pd.to_datetime(
    wizard1["date"], errors="coerce"
).dt.year

depth_ave = df_wizard_cols
depth_avg = [
    wizard1.loc[
        wizard1["year"] == yr, "depth"
    ].mean()
    for yr in years
]

# Build your 3‑column summary
summary_df2 = pd.DataFrame({
    "year": years,
    "avg_water_use": water_use_avg,
    "avg_depth": depth_avg
})


# In[703]:


import matplotlib.pyplot as plt

# summary_df should already exist with columns 'avg_water_use' and 'avg_depth'
plt.figure(figsize=(8, 6))
plt.scatter(summary_df2['avg_water_use'], summary_df2['avg_depth'])
plt.xlabel('Average Water Use')
plt.ylabel('Average Water Depth')
plt.title('Scatter Plot: Water Use HUC10 vs Water Depth')
plt.grid(True)
plt.show()

corr = summary_df2["avg_water_use"].corr(summary_df2["avg_depth"])
print("Correlation:", corr)


# In[ ]:




