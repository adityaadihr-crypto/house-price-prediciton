import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Dataset
data = {
    'area': [1000, 1500, 2000, 1200, 1800, 2200, 1400, 1600, 2100, 1700],
    'bedrooms': [2, 3, 4, 2, 3, 4, 3, 3, 4, 3],
    'bathrooms': [1, 2, 3, 1, 2, 3, 2, 2, 3, 2],
    'price': [2000000, 3500000, 5000000, 2500000, 4200000,
              5500000, 3000000, 3700000, 5200000, 4000000]
}
df = pd.DataFrame(data)

# Features & Target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results Table
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results['Actual (L)'] = (results['Actual']/100000).round(2).astype(str) + " L"
results['Predicted (L)'] = (results['Predicted']/100000).round(2).astype(str) + " L"
results['Error'] = abs(results['Actual'] - results['Predicted'])
final_table = results[['Actual (L)', 'Predicted (L)', 'Error']]

print("\n===== RESULTS =====\n")
print(final_table.to_string(index=False))

# Colorful Table
fig, ax = plt.subplots()
ax.axis('off')
table = ax.table(cellText=final_table.values,
                 colLabels=final_table.columns,
                 loc='center')

for j in range(len(final_table.columns)):
    table[(0, j)].set_facecolor('#4CAF50')
    table[(0, j)].set_text_props(color='white')

for i in range(len(final_table)):
    error = final_table.iloc[i]['Error']
    if error < 200000:
        color = '#A5D6A7'
    elif error < 500000:
        color = '#FFF59D'
    else:
        color = '#EF9A9A'
    for j in range(len(final_table.columns)):
        table[(i+1, j)].set_facecolor(color)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)
plt.title("House Price Prediction Table")
plt.savefig("colorful_table.png", bbox_inches='tight')
plt.show()

# Colorful Graph
plt.figure(figsize=(7, 6))
plt.scatter(results['Actual'], results['Predicted'], color='blue', s=100, label="Predictions")
min_val = min(results['Actual'].min(), results['Predicted'].min())
max_val = max(results['Actual'].max(), results['Predicted'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, linestyle='--', label="Perfect Prediction")
plt.xlabel("Actual Price", fontsize=12)
plt.ylabel("Predicted Price", fontsize=12)
plt.title("Actual vs Predicted Prices", fontsize=14)
plt.grid(True)
plt.legend()
plt.savefig("colorful_graph.png")
plt.show()