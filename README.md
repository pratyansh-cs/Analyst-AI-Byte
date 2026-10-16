# Analyst-AI-Byte
Byte task1
It is a dump and not chronological dumps just two commits


Setup -
1.Clone the Repository:
2.git clone https://github.com/pratyansh-cs/Analyst-AI-Byte.git


3.Install Dependencies:
   pip install smolagents pandas numpy matplotlib python-dotenv

4.Configure Environment Variables:
     Create a .env file in the root directory of your project and add your Gemini API key:
     GEMINI_API_KEY=your_actual_api_key_here

5.Add Your Dataset:   

    Ensure your target CSV file (e.g., flight_price.csv) is placed in the root directory alongside app.py and change the CSV_FILE name in app.py.

TOOLS-
describe_data(): Computes dataset dimensions, inspects column data types, and prints summary statistics.

check_quality(): Scans the dataset for data hygiene issues, counting missing values and duplicate rows.

group_summary(group_col, value_col, how): Performs aggregations (mean, sum, median, count, min, max) grouped by a target column.

top_correlations(): Identifies and ranks the strongest linear correlations between numeric columns.

quick_plot(kind, x, y): Generates and saves visual charts (bar, line, scatter, or hist) directly into an outputs/ folder.
