import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Set your directory
path = r"C:\Users\mdjon\OneDrive - Centria ammattikorkeakoulu Oy\Desktop\centria 26\ai\Helsinki_EV_Project"
os.chdir(path)

# --- PART 2: DATASET EVALUATION & CLEANING ---

# 1. Loading with latin-1 to handle Finnish characters (ä, ö)
try:
    df_pop = pd.read_csv('population dataset of helsinki_2018_2030.csv', sep=';', encoding='latin-1')
    df_vech = pd.read_csv('Vehicles in helsinki area.csv', low_memory=False, encoding='latin-1')
    df_stations = pd.read_csv('helsinki_ev_stations.csv', encoding='latin-1')
    print("All datasets loaded successfully!")
except Exception as e:
    print(f"Loading error: {e}")

# --- FIXING THE TYPEERROR (The '2030' column) ---
# We convert the target year column to numeric, forcing errors to 'NaN' (Not a Number)
df_pop['2030'] = pd.to_numeric(df_pop['2030'], errors='coerce')

# --- FIXING THE COLUMN NAMES ---
# Because of encoding, 'Ikä/Age' might show up as 'IkÃ¤/Age'. Let's rename it to be safe.
df_pop = df_pop.rename(columns={df_pop.columns[4]: 'Age_Group'})

# 2. Filtering Population Data
# We look for the 'Total' (Yhteensä) rows to get district totals
# We use .str.contains with 'Yhteen' to be safe with encoding variations
df_pop_total = df_pop[df_pop['Age_Group'].str.contains('Yhteen', na=False)].copy()

# Remove the overall city total so we only see individual districts
# (Assuming '091 Helsinki' is the first row and represents the whole city)
df_districts = df_pop_total[df_pop_total['Alue/District'] != '091 Helsinki']

# --- DATA EXPLORATION INSIGHTS ---

# Now nlargest will work because '2030' is a numeric type
top_growth = df_districts.nlargest(5, '2030')[['Alue/District', '2030']]
print("\nTop 5 Districts by 2030 Population Forecast:\n", top_growth)

# Process Station Data
df_stations['Plug_Count'] = df_stations['Connectors'].str.extract('(\d+)').astype(float)


# --- STEP: TRANSLATE DISTRICT NAMES ---
# Create a dictionary for mapping Finnish to English
translations = {
    '091 1 Eteläinen suurpiiri': 'Southern District',
    '091 2 Läntinen suurpiiri': 'Western District',
    '091 7 Itäinen suurpiiri': 'Eastern District',
    '091 3 Keskinen suurpiiri': 'Central District',
    '091 5 Koillinen suurpiiri': 'Northeastern District',
    '091 4 Pohjoinen suurpiiri': 'Northern District',
    '091 6 Kaakkoinen suurpiiri': 'Southeastern District',
    '091 8 Östersundomin suurpiiri': 'Östersundom District'
}

# Apply the translation to the DataFrame
df_districts['Alue/District_EN'] = df_districts['Alue/District'].map(translations)

# Update the top_growth calculation to use the English names
top_growth_en = df_districts.nlargest(5, '2030')[['Alue/District_EN', '2030']]

print("\nTop 5 Districts (English):\n", top_growth_en)

# --- UPDATED VISUALIZATION ---
plt.figure(figsize=(10,6))
plt.bar(top_growth_en['Alue/District_EN'], top_growth_en['2030'], color='teal')
plt.title('Future EV Demand: Top 5 Districts by 2030 Population')
plt.ylabel('Projected Population')
plt.show()




# Plot 1: Future Demand (District Population in 2030)
plt.figure(figsize=(10,6))
plt.bar(top_growth['Alue/District'], top_growth['2030'], color='teal')
plt.title('Future AI Targeting: Highest Population Districts (2030)')
plt.ylabel('Projected Population')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot 2: Identifying Current Leading Operators
plt.figure(figsize=(10,6))
df_stations.groupby('Operator')['Plug_Count'].sum().sort_values(ascending=False).head(5).plot(kind='bar', color='orange')
plt.title('Current Infrastructure: Total Plugs by Operator')
plt.ylabel('Number of Plugs')
plt.tight_layout()
plt.show()